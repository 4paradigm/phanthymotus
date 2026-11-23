// Optional SensorData recovery using the pinned RTAB-Map algorithms.
// Coordinates: native_odom <- visual_odom / external_odom. No robot/frame policy.
#pragma once
#include "WallAlignment.h"
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/Memory.h>
#include <rtabmap/core/Signature.h>
#include <rtabmap/core/Graph.h>
#include <rtabmap/core/Odometry.h>
#include <rtabmap/core/OdometryInfo.h>
#include <rtabmap/core/RegistrationIcp.h>
#include <rtabmap/core/RegistrationInfo.h>
#include <rtabmap/core/Compression.h>
#include <rtabmap/utilite/ULogger.h>
#include <memory>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <optional>
#include <set>

namespace motus {
class VisualRecovery
{
    using ExternalRun=std::array<unsigned char,16>;

public:
    enum State { External=0, Searching=1, Localized=2, VisualTracking=3, InertialRegistration=4, ScanTracking=5 };
    explicit VisualRecovery(const rtabmap::ParametersMap & parameters, bool rgbdImageRecovery=false):
        neighborIcp_(parameters), state_(External), missing_(false), waiting_(false), aligned_(false),
        localizationMode_(false), relocalizations_(0), visualInliers_(0), imageMode_(false), initialized_(false),
        recoveryNode_(0), recoveryAnchor_(0), recoveryInliers_(0), rgbdImageRecovery_(rgbdImageRecovery)
    {
        auto p=parameters;
        p["Reg/Strategy"]="0"; // OdometryF2M visual geometry, not scan-only ICP.
        p["Odom/Strategy"]="0"; p["Odom/ResetCountdown"]="0";
        p["OdomF2M/MaxSize"]="2000";
        estimator_.reset(rtabmap::Odometry::create(p));
        auto registration=parameters;
        registration["Icp/PointToPlane"]="true";
        registration["Icp/PointToPlaneMinComplexity"]="0.02";
        registration["Icp/PointToPlaneLowComplexityStrategy"]="0";
        inertialIcp_.reset(new rtabmap::RegistrationIcp(registration));
        // Preserve fine structure for the stricter two-way older-anchor check.
        auto revisitRegistration=registration;
        float voxelSize=rtabmap::Parameters::defaultIcpVoxelSize();
        rtabmap::Parameters::parse(registration,"Icp/VoxelSize",voxelSize);
        if(voxelSize>.06f) revisitRegistration["Icp/VoxelSize"]="0.06";
        revisitIcp_.reset(new rtabmap::RegistrationIcp(revisitRegistration));
        // Scan tracking starts without a motion guess, so allow a little more translation.
        registration["Icp/MaxTranslation"]="0.25";
        registration["Icp/CorrespondenceRatio"]="0.07";
        scanIcp_.reset(new rtabmap::RegistrationIcp(registration));
        externalToNative_=rtabmap::Transform::getIdentity();
        bool refining=rtabmap::Parameters::defaultRGBDNeighborLinkRefining();
        rtabmap::Parameters::parse(p,"RGBD/NeighborLinkRefining",refining);
        imageMode_=!refining;
        rtabmap::Parameters::parse(p, "Vis/MinInliers", minInliers_=rtabmap::Parameters::defaultVisMinInliers());
        rtabmap::Parameters::parse(p, "RGBD/OptimizeMaxError", maxError_=rtabmap::Parameters::defaultRGBDOptimizeMaxError());
        rtabmap::Parameters::parse(p, "RGBD/MaxOdomCacheSize", maxOdomCacheSize_=rtabmap::Parameters::defaultRGBDMaxOdomCacheSize());
    }
    int state() const {return state_;}
    int relocalizations() const {return relocalizations_;}
    int visualInliers() const {return visualInliers_;}
    int recoveryNode() const {return recoveryNode_;}
    int recoveryAnchor() const {return recoveryAnchor_;}
    int recoveryInliers() const {return recoveryInliers_;}
    const rtabmap::Transform & odom() const {return odom_;}

    static void finalize(rtabmap::Rtabmap & map)
    {
        std::map<int,rtabmap::Transform> poses;
        std::multimap<int,rtabmap::Link> links;
        const auto &local=map.getLocalOptimizedPoses();
        map.getGraph(poses,links,false,true); // Connected IDs, including LTM; never use raw poses.
        float bound=rtabmap::Parameters::defaultRGBDOptimizeMaxError();
        rtabmap::Parameters::parse(map.getParameters(),"RGBD/OptimizeMaxError",bound);
        auto valid=[&](const std::map<int,rtabmap::Transform> & p,
                       const std::multimap<int,rtabmap::Link> & l)
        {
            float linear=0,angular=0,linearError=0,angularError=0;
            rtabmap::graph::computeMaxGraphErrors(p,l,linear,angular,linearError,angularError);
            return !p.empty() && std::isfinite(linear) && std::isfinite(angular) &&
                (bound<=0 || (linear<=bound && angular<=bound));
        };
        if(std::all_of(poses.begin(),poses.end(),[&](const auto &p){return local.count(p.first);}) &&
           valid(local,map.getLocalConstraints()))
            return; // Saving an already complete, validated native graph must not move it.
        const auto completeSize=poses.size();
        map.getGraph(poses,links,true,true); // Native global optimization, no sensor blobs.
        if(valid(poses,links) && (local.empty() || poses.count(local.begin()->first)))
            map.setOptimizedPoses(poses,links);
        else
            UWARN("Global finalization failed native graph validation; retaining %zu/%zu cached poses. Complete saved graph is unverified.",local.size(),completeSize);
    }

    bool process(rtabmap::Rtabmap & map, rtabmap::SensorData & data,
                 const rtabmap::Transform & external, const cv::Mat & covariance,
                 const std::vector<float> & velocity=std::vector<float>(),
                 const std::map<std::string,float> & statistics=std::map<std::string,float>())
    {
        recoveryNode_=recoveryAnchor_=recoveryInliers_=0;
        if(visualLossStamp_>0. && !(data.stamp()>visualLossStamp_))
        {
            walls_.reset();
            // Reject delayed observations before they can alter F2M. A clock
            // regression also invalidates the pending same-gauge recovery.
            visualLossStamp_=0.; visualAlignment_.reset(); visualInliers_=0;
            state_=Searching; odom_.setNull(); return false;
        }
        const auto chassis=takePayload(data,"MOTUSCH1",40);
        if(!chassis.empty()) sawChassis_=true;
        if(!takePayload(data,"MOTUSRAW",8).empty())
        {
            walls_.reset();
            // Legacy scan-only inputs have no declared gauge. Once this stream
            // declares one, an unmarked pose cannot move the established map.
            if(sawChassis_ || external.isNull())
            {
                state_=Searching; odom_.setNull(); return false;
            }
            state_=External; odom_=external;
            const bool accepted=map.process(data,external,covariance,velocity,statistics);
            if(accepted) rememberScan(map,data);
            return accepted;
        }
        auto externalRun=takeExternalRun(data);
        if(!chassis.empty())
        {
            ExternalRun run;
            std::memcpy(run.data(),chassis.data+24,run.size());
            if(run!=ExternalRun{}) externalRun=run;
        }
        const auto originPacket=chassis.empty()?takePayload(data,"MOTUSOR1",24):chassis;
        const bool declaresOrigin=!originPacket.empty() && !external.isNull();
        const bool images=!data.imageRaw().empty() && !data.depthOrRightRaw().empty();
        bool sameOrigin=false;
        if(declaresOrigin)
        {
            ExternalRun origin;
            std::memcpy(origin.data(),originPacket.data+8,origin.size());
            sameOrigin=externalOrigin_ && origin==*externalOrigin_;
            if((externalOrigin_ || !chassis.empty()) && !sameOrigin && !map.getLocalOptimizedPoses().empty())
            {
                walls_.reset();
                // A new odometry gauge needs an independent map constraint.
                state_=Searching; odom_.setNull(); visualInliers_=0;
                waiting_=true;
                const bool scanLocalized=relocalizeScan(map,data,statistics,externalRun && !chassis.empty());
                if(!scanLocalized)
                {
                    if(!images) return false;
                    const auto previousScanAnchor=scanAnchor_;
                    const auto result=relocalizeImages(map,data,covariance,statistics);
                    if(result!=ImageQueryResult::Localized)
                    {
                        // A rejected query must preserve the last measured scan witness.
                        scanAnchor_=previousScanAnchor;
                        return result==ImageQueryResult::Unverified;
                    }
                }
                rtabmap::OdometryInfo info;
                finishLocalization(map,data,external,rtabmap::Transform(),info,false,!scanLocalized,{});
                externalRun_=externalRun;
                lastAcceptedExternalRun_=externalRun;
                previousExternal_=external;
                previousExternalStamp_=data.stamp();
                initialized_=true;
                externalOrigin_=origin;
                return true;
            }
            externalOrigin_=origin;
        }
        const double externalAge=data.stamp()-previousExternalStamp_;
        const double outageAge=data.stamp()-chassisOutageStart_;
        const bool recentOutage=chassisOutageStart_>0. && outageAge>=0. && outageAge<=2.;
        const bool witnessedRun=externalRun && externalRun==externalRun_;
        // Dense verified poses can keep arriving between sparse native scans.
        // Keep the aligned run across a short scan outage, measured from its
        // last accepted external pose. A declared origin alone is weaker.
        // The gateway carries the run token through a visual pause only when
        // the runtime measured the same F2M landmarks within 4.5 s of its last
        // pose; allow the sparse native scan cadence on top of that bound.
        const bool witnessedGap=externalRun_ && externalAge>0. && externalAge<=maxWitnessedGap_;
        // A continuously measured stream can outlive slow scan delivery.
        // Apply the longer scan gap only before any unverified outage.
        const double scanGap=witnessedRun && !waiting_ && !missing_?
            (chassis.empty()?maxLiveScanGap_:externalAge):maxExternalScanGap_;
        const bool continuousExternal=(witnessedRun || (chassis.empty() && sameOrigin)) &&
            (!waiting_ || recentOutage || (witnessedRun && witnessedGap)) &&
            !localizationMode_ && continuousMotion(external,data.stamp(),scanGap);
        if(!continuousExternal) walls_.reset();
        const bool waitForChassis=external.isNull() && (externalRun_ || externalOrigin_) && !localizationMode_ &&
            externalAge>0. && externalAge<=maxExternalScanGap_ &&
            (!chassisOutageStart_ || recentOutage || witnessedGap) &&
            !map.getLocalOptimizedPoses().empty();
        if(waitForChassis && !chassisOutageStart_) chassisOutageStart_=data.stamp();
        if(!waitForChassis) chassisOutageStart_=0.;
        if(!continuousExternal && !waitForChassis) externalRun_.reset();
        rtabmap::Transform guess;
        if(external.isNull() && takePrediction(data,guess))
        {
            const auto registered=registerPrediction(map,data,guess,statistics);
            if(registered.has_value())
            {
                chassisOutageStart_=0.; externalRun_.reset();
                return *registered;
            }
        }
        rtabmap::OdometryInfo info;
        rtabmap::Transform visual;
        if(!initialized_)
        {
            waiting_=!map.getLocalOptimizedPoses().empty() ||
                (map.getMemory() && map.getMemory()->getLastWorkingSignature());
            if(!waiting_ && external.isNull())
            {
                // An empty database has no anchor to localize against. Wait for
                // the first measured origin before entering localization mode.
                state_=Searching;
                return false;
            }
            initialized_=true;
        }
        if(images!=imageMode_)
        {
            // Keep the supplied motion covariance. Scan ICP on a weakly
            // constrained surface can replace it with an overconfident edge.
            // Visual candidates still require native visual geometry AND ICP.
            map.parseParameters({{"RGBD/NeighborLinkRefining","false"},
                                 {"RGBD/LoopCovLimited","true"},
                                 {"RGBD/ProximityOdomGuess","false"}});
            imageMode_=images;
        }
        const auto framesBefore=estimator_->framesProcessed();
        if(images) visual=processVisual(data,info);
        visualInliers_=info.reg.inliers;
        const bool tracked=!visual.isNull() && !info.lost &&
            visualInliers_>=minInliers_ && validCovariance(info.reg.covariance);
        const bool visualSeed=!visual.isNull() && !info.lost && info.keyFrameAdded &&
            estimator_->framesProcessed()==1 && info.localMapSize>=minInliers_ && cv::checkRange(visual.dataMatrix());
        if(images && !tracked)
        {
            lastVisual_.setNull(); lastNative_.setNull();
        }
        if(external.isNull() && !missing_)
        {
            if(!waiting_ && !localizationMode_ && !tracked && visualInliers_>0 && !map.getLocalOptimizedPoses().empty())
            {
                const auto *anchor=map.getMemory()->getLastWorkingSignature();
                const double age=anchor?data.stamp()-anchor->getStamp():0.;
                if(age>0. && age<=4.) externalGraceStart_=data.stamp();
            }
            waiting_=true;
            const auto alignment=lastVisual_.isNull()?initialVisualToNative_:lastNative_*lastVisual_.inverse();
            aligned_=!alignment.isNull();
            if(aligned_) visualToNative_=alignment;
        }
        if(images || external.isNull()) initialVisualToNative_.setNull();
        missing_=external.isNull();
        int reacquired=0;
        if(!localizationMode_ && visualAlignment_ &&
           (visualLossStamp_>0. || (images && !tracked)))
        {
            const auto &anchor=*visualAlignment_;
            const double age=data.stamp()-anchor.stamp;
            const bool sameSequence=age>0. && age<=2. && framesBefore==anchor.frames &&
                estimator_->framesProcessed()==framesBefore+(tracked?1u:0u);
            if(waiting_ && !continuousExternal) aligned_=false;
            if(sameSequence && !tracked)
            {
                // Keep the old landmarks while independently testing other recovery sources.
                visualLossStamp_=data.stamp();
            }
            else
            {
                visualLossStamp_=0.;
                if(waiting_ && !continuousExternal && sameSequence && tracked && cv::checkRange(visual.dataMatrix()) &&
                   plausibleMotion(anchor.visual.inverse()*visual,age))
                    reacquired=countOldVisualInliers(info,anchor.landmarks);
                if(reacquired>=minInliers_)
                {
                    visualToNative_=anchor.native*anchor.visual.inverse();
                    aligned_=true;
                    externalGraceStart_=0.;
                }
                else
                    visualAlignment_.reset();
            }
        }
        const bool recoveredVisual=reacquired>=minInliers_;
        std::map<std::string,float> reacquisitionStats;
        if(recoveredVisual)
        {
            reacquisitionStats=statistics;
            reacquisitionStats["Motus/VisualReacquiredInliers/"]=static_cast<float>(reacquired);
        }
        const auto &acceptedStats=reacquisitionStats.empty()?statistics:reacquisitionStats;
        // A failed measured image can revoke VO alignment. A scan without
        // images leaves the visual estimator's established gauge unchanged.
        if(waiting_ && aligned_ && !tracked && !continuousExternal && images)
            aligned_=false;
        if(waiting_ && aligned_ && !external.isNull() && tracked && !recoveredVisual)
        {
            externalToNative_=visualToNative_*visual*external.inverse();
            waiting_=false;
        }
        const auto verifiesNeighbor=[&](const rtabmap::Transform & candidate, bool requireAgreement=false)
        {
            const auto *last=map.getMemory()->getLastWorkingSignature();
            if(!last) return !requireAgreement;
            const auto expected=last->getPose().inverse()*candidate;
            const auto near=[](const rtabmap::Transform & relative)
            {
                return relative.getNorm()<=.1 && relative.getAngle(rtabmap::Transform::getIdentity())<=5.*CV_PI/180.;
            };
            if(requireAgreement && !near(expected)) return false;
            const auto measured=neighborIcp_.computeTransformation(last->sensorData(),data,expected);
            return !measured.isNull() && (!requireAgreement || near(expected.inverse()*measured));
        };
        const double graceAge=data.stamp()-externalGraceStart_;
        const bool withinGrace=externalGraceStart_>0. && graceAge>=0. && graceAge<=2.;
        const bool verifiedRetry=waiting_ && !continuousExternal && !recoveredVisual && !external.isNull() &&
            (withinGrace || retryExternalNeighbor_ || visualLossStamp_>0.) &&
            verifiesNeighbor(externalToNative_*external,withinGrace || visualLossStamp_>0.);
        if(verifiedRetry || continuousExternal)
            waiting_=false;
        const double scanAge=scanAnchor_?data.stamp()-scanAnchor_->scan.stamp():0.;
        const double scanRunAge=scanAnchor_?data.stamp()-scanAnchor_->referenceStamp:0.;
        if((external.isNull() || waiting_) && !(aligned_ && tracked) &&
           !localizationMode_ && scanAge>0. && scanRunAge<=4.)
        {
            if(registerScan(map,data,acceptedStats,scanAge,waitForChassis))
            {
                if(tracked)
                {
                    visualToNative_=odom_*visual.inverse();
                    aligned_=true;
                    rememberVisualAlignment(data.stamp(),visual,info);
                }
                externalGraceStart_=0.;
                return true;
            }
            if(scanAnchor_->recovered)
            {
                // A failed registration publishes no pose. A later scan can
                // still verify motion before the last measured anchor expires.
                state_=Searching; odom_.setNull();
                return false;
            }
        }
        if(waiting_ && aligned_ && !tracked && !continuousExternal && !images &&
           scanAnchor_ && scanRunAge<=4.)
        {
            state_=Searching; odom_.setNull();
            return false;
        }
        if(withinGrace && external.isNull() && (!images || info.features>0))
        {
            // A weak observation cannot move the map. Give the measured stream
            // time to return before RTAB-Map saves and reloads for localization.
            state_=Searching; odom_.setNull(); return false;
        }
        externalGraceStart_=0.;
        if(external.isNull())
            retryExternalNeighbor_=false;
        if(!waiting_ && !external.isNull())
        {
            retryExternalNeighbor_=false;
            state_=External; odom_=externalToNative_*external;
            if(tracked && !lastVisual_.isNull())
            {
                const auto predicted=lastNative_*lastVisual_.inverse()*visual;
                const auto residual=predicted.inverse()*odom_;
                double linear=0.,angular=0.;
                for(int i=0;i<6;++i)
                {
                    const double variance=info.reg.covariance.at<double>(i,i)+
                        (validCovariance(covariance)?2.*covariance.at<double>(i,i):2.);
                    (i<3?linear:angular)=std::max(i<3?linear:angular,variance);
                }
                if(maxError_>0 && (residual.getNorm()>maxError_*std::sqrt(linear) ||
                   residual.getAngle(rtabmap::Transform::getIdentity())>maxError_*std::sqrt(angular)))
                {
                    // A numerical origin jump is tested against independent VO,
                    // never a producer's lost/segment/frame labels.
                    externalToNative_=predicted*external.inverse(); odom_=predicted;
                }
            }
            if(!continuousExternal && !verifiedRetry && (!tracked || lastVisual_.isNull()))
            {
                // Validate the supplied constraint; never replace it with an ICP pose.
                // Memory::computeIcpTransform uses the stricter merged-scan overlap.
                if(!verifiesNeighbor(odom_))
                {
                    waiting_=true; aligned_=!lastVisual_.isNull();
                    // A later scan can still verify a continuing external
                    // origin. It cannot reuse the failed neighbor constraint.
                    retryExternalNeighbor_=true;
                    if(aligned_) visualToNative_=lastNative_*lastVisual_.inverse();
                    state_=Searching; odom_.setNull(); return false;
                }
            }
            if(tracked)
            {
                lastVisual_=visual; lastNative_=odom_;
            }
            const auto processExternal=[&]()
            {
                bool accepted;
                if(continuousExternal)
                {
                    // A verified stream can turn past the camera's field of view.
                    // Its measured pose defines a neighbor edge, with no added loop edge.
                    cv::Mat motionCovariance=validCovariance(covariance)?covariance.clone():cv::Mat::eye(6,6,CV_64F);
                    constexpr double positionVariance=.05*.05, angleVariance=(3.*CV_PI/180.)*(3.*CV_PI/180.);
                    for(int i=0;i<6;++i)
                        motionCovariance.at<double>(i,i)=std::max(motionCovariance.at<double>(i,i),
                            i<3?positionVariance:angleVariance);
                    auto stats=acceptedStats;
                    stats["Motus/MeasuredExternalContinuity/"]=1.f;
                    accepted=map.process(data,odom_,motionCovariance,velocity,stats);
                }
                else accepted=map.process(data,odom_,covariance,velocity,acceptedStats);
                if(accepted) rememberScan(map,data);
                if(accepted && continuousExternal) walls_.observe(map,data);
                // Chassis-only growth preserves any pending visual recovery.
                if(accepted && tracked) rememberVisualAlignment(data.stamp(),visual,info);
                // A mature first keyframe establishes a gauge, but doesn't yet measure motion.
                // Only a later tracked frame may use this measured anchor during external loss.
                if(accepted && visualSeed) initialVisualToNative_=odom_*visual.inverse();
                if(accepted && (externalRun || declaresOrigin))
                {
                    externalRun_=externalRun;
                    lastAcceptedExternalRun_=externalRun;
                    previousExternal_=external;
                    previousExternalStamp_=data.stamp();
                }
                return accepted;
            };
            const auto *last=images?map.getMemory()->getLastWorkingSignature():nullptr;
            if(last && map.getLocalOptimizedPoses().count(last->id()) &&
               !hasVisualGeometry(last->getWords3()) && hasVisualGeometry(data.keypoints3D()))
            {
                // Retain a visual anchor after scan-only keyframes, even if
                // the first usable image arrives while the robot is stationary.
                rtabmap::ParametersMap restore;
                for(const auto *key:{"RGBD/LinearUpdate","RGBD/AngularUpdate"})
                {
                    const auto value=map.getParameters().find(key);
                    restore[key]=value==map.getParameters().end()?
                        rtabmap::Parameters::getDefaultParameters().at(key):value->second;
                }
                map.parseParameters({{"RGBD/LinearUpdate","0"},{"RGBD/AngularUpdate","0"}});
                const bool accepted=processExternal();
                map.parseParameters(restore);
                return accepted;
            }
            return processExternal();
        }
        if(aligned_ && tracked)
        {
            state_=VisualTracking; odom_=visualToNative_*visual;
            lastVisual_=visual; lastNative_=odom_;
            cv::Mat motionCovariance=info.reg.covariance.clone();
            // Native registration alone omits uncertainty from the projected
            // depth and RGB/scan time alignment used by this mapping input.
            constexpr double positionVariance=.02*.02, angleVariance=(CV_PI/180.)*(CV_PI/180.);
            for(int i=0;i<6;++i)
                motionCovariance.at<double>(i,i)=std::max(motionCovariance.at<double>(i,i),
                    i<3?positionVariance:angleVariance);
            const bool accepted=map.process(data,odom_,motionCovariance,std::vector<float>(),acceptedStats);
            if(accepted)
            {
                retryExternalNeighbor_=false;
                rememberVisualAlignment(data.stamp(),visual,info);
                rememberScan(map,data);
            }
            return accepted;
        }
        state_=Searching; odom_.setNull();
        // A short absence does not revoke the last independently witnessed
        // chassis run. Only a new measured pose carrying that same run may resume it.
        if(waitForChassis || visualLossStamp_>0.) return false;
        const auto recoverGeometry=[&]()
        {
            const bool sameGaugeRunChange=sameOrigin && !chassis.empty() && externalRun &&
                lastAcceptedExternalRun_ && externalRun!=lastAcceptedExternalRun_ &&
                continuousMotion(external,data.stamp(),maxLiveScanGap_);
            const auto guess=sameGaugeRunChange?externalToNative_*external:rtabmap::Transform();
            if(!relocalizeScan(map,data,statistics,false,guess)) return false;
            finishLocalization(map,data,external,visual,info,tracked,images,externalRun);
            return true;
        };
        if(!images) return recoverGeometry();
        const auto queryResult=relocalizeImages(map,data,covariance,statistics);
        if(queryResult==ImageQueryResult::Rejected) return false;
        if(queryResult!=ImageQueryResult::Localized)
            return recoverGeometry() || queryResult==ImageQueryResult::Unverified;
        finishLocalization(map,data,external,visual,info,tracked,images,externalRun);
        return true;
    }

private:
    enum class ImageQueryResult { NotProcessed, Unverified, Rejected, Localized };

    ImageQueryResult relocalizeImages(rtabmap::Rtabmap & map, rtabmap::SensorData & data,
                                     const cv::Mat & covariance,
                                     const std::map<std::string,float> & statistics)
    {
        beginLocalization(map);
        // Failed localization can clear RTAB-Map's optimized cache when there
        // is no previous localization node. Keep the last accepted graph.
        const auto previousPoses=map.getLocalOptimizedPoses();
        const auto previousLinks=map.getLocalConstraints();
        const auto processQuery=[&](){return map.process(data,rtabmap::Transform(),covariance,std::vector<float>(),statistics);};
        bool processed;
        if(rgbdImageRecovery_ && !sawChassis_)
        {
            // Archived words can be incompatible with the query even when both
            // retained RGB-D views register. Recompute only for localization so
            // mapping continues to retain 3D words and descriptors on new nodes.
            const auto & parameters=map.getParameters();
            const auto key=rtabmap::Parameters::kRGBDLoopClosureReextractFeatures();
            const auto option=parameters.find(key);
            const rtabmap::ParametersMap restore{{key,option==parameters.end()?
                rtabmap::Parameters::getDefaultParameters().at(key):option->second}};
            map.parseParameters({{key,"true"}});
            try
            {
                processed=processQuery();
            }
            catch(...)
            {
                map.parseParameters(restore);
                throw;
            }
            map.parseParameters(restore);
        }
        else processed=processQuery();
        const auto &stats=map.getStatistics();
        // Native loop id is emitted only after registration and graph validation.
        // Proximity and appearance hypotheses alone cannot authorize recovery.
        auto inliers=stats.data().find(rtabmap::Statistics::kLoopVisual_inliers());
        if(!processed || stats.loopClosureId()<=0 ||
           inliers==stats.data().end() || inliers->second<minInliers_ ||
           !validCovariance(stats.localizationCovariance()) || map.getLastLocalizationPose().isNull())
        {
            map.setOptimizedPoses(previousPoses,previousLinks);
            return processed?ImageQueryResult::Unverified:ImageQueryResult::NotProcessed;
        }

        const int anchor=stats.loopClosureId(), query=stats.refImageId();
        const int localizationInliers=static_cast<int>(inliers->second);
        const auto location=map.getLastLocalizationPose();
        const auto & observations=stats.getSignaturesData();
        const auto observation=observations.find(query);
        rtabmap::Link measured;
        if(observation!=observations.end())
        {
            const auto & edges=observation->second.getLinks();
            const auto edge=edges.find(anchor); // Signature links are keyed by destination.
            if(edge!=edges.end() && edge->second.type()==rtabmap::Link::kGlobalClosure)
                measured=edge->second.inverse();
        }
        const auto locCovariance=stats.localizationCovariance().clone();
        auto poses=map.getLocalOptimizedPoses();
        auto links=map.getLocalConstraints();
        poses.erase(query);
        for(auto i=links.begin();i!=links.end();)
            if(i->second.from()==query || i->second.to()==query) i=links.erase(i); else ++i;
        if(!measured.isValid() || poses.find(anchor)==poses.end())
        {
            map.setOptimizedPoses(previousPoses,previousLinks);
            return ImageQueryResult::Unverified;
        }
        if(!promote(map,data,measured,location,locCovariance,statistics,poses,links,previousPoses,previousLinks))
            return ImageQueryResult::Rejected;
        recoveryInliers_=localizationInliers;
        return ImageQueryResult::Localized;
    }

    void beginLocalization(rtabmap::Rtabmap & map)
    {
        if(localizationMode_) return;
        const auto &parameters=map.getParameters();
        const auto proximity=parameters.find("RGBD/ProximityBySpace");
        proximityBeforeLocalization_=proximity==parameters.end()?
            rtabmap::Parameters::getDefaultParameters().at("RGBD/ProximityBySpace"):proximity->second;
        retryExternalNeighbor_=false;
        visualLossStamp_=0.;
        visualAlignment_.reset();
        scanAnchor_.reset();
        // A missing odometry guess cannot justify proximity localization.
        map.parseParameters({{"Mem/IncrementalMemory","false"}, {"RGBD/ProximityBySpace","false"},
                             {"RGBD/MaxOdomCacheSize","0"}});
        localizationMode_=true;
    }

    bool promote(rtabmap::Rtabmap & map, rtabmap::SensorData & data, const rtabmap::Link & measured,
                 const rtabmap::Transform & location, const cv::Mat & covariance,
                 const std::map<std::string,float> & statistics, std::map<int,rtabmap::Transform> & poses,
                 const std::multimap<int,rtabmap::Link> & links,
                 const std::map<int,rtabmap::Transform> & previousPoses,
                 const std::multimap<int,rtabmap::Link> & previousLinks)
    {
        const int anchor=measured.from();
        // Pinned parseParameters clears optimized poses when returning to mapping.
        // Insert the verified observation as a leaf. A second, opportunistic
        // constraint here can move old poses or replace the measured closure.
        rtabmap::ParametersMap restore;
        rtabmap::ParametersMap insertion{{"Rtabmap/LoopThr","1.01"},
            {"RGBD/AggressiveLoopThr","1.01"}, {"RGBD/ProximityByTime","false"}};
        for(const auto & param:insertion)
        {
            const auto & current=map.getParameters();
            const auto value=current.find(param.first);
            restore[param.first]=value!=current.end()?value->second:
                rtabmap::Parameters::getDefaultParameters().at(param.first);
        }
        restore["RGBD/ProximityBySpace"]=proximityBeforeLocalization_;
        insertion["Mem/IncrementalMemory"]="true";
        insertion["RGBD/MaxOdomCacheSize"]=std::to_string(maxOdomCacheSize_);
        map.parseParameters(insertion); // Spatial proximity is already off while searching.
        localizationMode_=false;
        map.setOptimizedPoses(poses,links);
        odom_=location;
        const bool promoted=map.process(data,odom_,covariance,std::vector<float>(),statistics);
        map.parseParameters(restore);
        const int node=map.getLastLocationId();
        const auto *signature=map.getMemory()->getSignature(node);
        const bool inserted=promoted && signature && map.isInSTM(node) && !poses.count(node) &&
            signature->getStamp()==data.stamp();
        bool connected=false;
        if(inserted && signature->getLinks().empty() && signature->getLandmarks().empty())
        {
            poses.emplace(node,location);
            map.setOptimizedPoses(poses,links);
            connected=map.addLink(rtabmap::Link(anchor,node,rtabmap::Link::kGlobalClosure,
                measured.transform(),measured.infMatrix()),true) &&
                signature->hasLink(anchor,rtabmap::Link::kGlobalClosure);
        }
        if(!connected)
        {
            // Native process may discard the current observation itself.
            if(inserted) map.deleteLastLocation();
            beginLocalization(map);
            map.setOptimizedPoses(previousPoses,previousLinks);
            odom_.setNull(); aligned_=false; return false;
        }
        recoveryNode_=node; recoveryAnchor_=anchor;
        return true;
    }

    struct ScanMatch
    {
        int anchor;
        rtabmap::Transform relative;
        rtabmap::RegistrationInfo result;
    };

    std::optional<ScanMatch> matchEarlierScan(rtabmap::Rtabmap & map, rtabmap::SensorData & data,
            const std::map<int,rtabmap::Transform> & poses,const rtabmap::Transform & guess)
    {
        const auto identity=rtabmap::Transform::getIdentity();
        const auto nearby=[](const rtabmap::Transform & delta,double meters,double degrees)
        {
            return !delta.isNull() && delta.getNorm()<=meters &&
                delta.getAngle(rtabmap::Transform::getIdentity())<=degrees*CV_PI/180.;
        };
        const auto supported=[](const rtabmap::RegistrationInfo & info)
        {
            return info.icpStructuralComplexity>=.02 && info.icpCorrespondences>=1000 &&
                info.icpInliersRatio>=minScanRelocalizationOverlap_ && validCovariance(info.covariance);
        };
        std::multimap<double,int> candidates;
        auto *memory=map.getMemory();
        for(const auto &pose:poses)
        {
            const auto *signature=memory->getSignature(pose.first);
            if(pose.first==scanMapAnchor_ || !signature || signature->getPose().isNull()) continue;
            const auto delta=signature->getPose().inverse()*guess;
            const double cost=delta.getNorm()+delta.getAngle(identity);
            if(std::isfinite(cost)) candidates.emplace(cost,pose.first);
        }
        int attempts=0;
        for(const auto &candidate:candidates)
        {
            if(++attempts>3) break;
            const int anchor=candidate.second;
            rtabmap::LaserScan scan;
            memory->getNodeData(anchor,false,true,false,false).uncompressDataConst(nullptr,nullptr,&scan);
            if(scan.empty()) continue;
            rtabmap::SensorData reference;
            reference.setLaserScan(scan);
            // The raw keyframe pose and fused pose share an odometry gauge.
            // Optimized poses are used only to place an independently verified edge.
            const auto initial=memory->getSignature(anchor)->getPose().inverse()*guess;
            rtabmap::RegistrationInfo forward,reverse;
            const auto relative=revisitIcp_->computeTransformation(reference,data,initial,&forward);
            if(relative.isNull() || !supported(forward) || !nearby(initial.inverse()*relative,.10,5.)) continue;
            const auto backward=revisitIcp_->computeTransformation(data,reference,initial.inverse(),&reverse);
            if(!backward.isNull() && supported(reverse) && nearby(relative*backward,.02,1.))
                return ScanMatch{anchor,relative,std::move(forward)};
        }
        return std::nullopt;
    }

    bool relocalizeScan(rtabmap::Rtabmap & map, rtabmap::SensorData & data,
                        const std::map<std::string,float> & statistics,bool chassisReset=false,
                        const rtabmap::Transform & sameGaugeGuess=rtabmap::Transform())
    {
        auto poses=map.getLocalOptimizedPoses();
        if(poses.empty() || data.laserScanRaw().empty()) return false;
        // Short scan bridges can create nodes of their own. Their correspondences
        // do not advance the independent map anchor used for re-localization.
        if(!scanMapAnchor_) scanMapAnchor_=poses.rbegin()->first;
        const auto anchorPose=poses.find(scanMapAnchor_);
        if(anchorPose==poses.end()) return false;
        rtabmap::LaserScan scan;
        map.getMemory()->getNodeData(scanMapAnchor_,false,true,false,false)
            .uncompressDataConst(nullptr,nullptr,&scan);
        if(scan.empty()) return false;
        rtabmap::SensorData reference;
        reference.setLaserScan(scan);
        rtabmap::RegistrationInfo result;
        const auto registerFrom=[&](const rtabmap::Transform & guess)
        {
            result=rtabmap::RegistrationInfo();
            const auto relative=inertialIcp_->computeTransformation(reference,data,guess,&result);
            if(relative.isNull() || result.icpStructuralComplexity<.02 || result.icpCorrespondences<200 ||
               !validCovariance(result.covariance)) return rtabmap::Transform();
            if(result.icpInliersRatio>=minScanRelocalizationOverlap_) return relative;
            // A recent chassis reset at a stationary viewpoint may have partial
            // overlap. Require a reciprocal match and a second measured scan
            // distinct from the stored keyframe, all agreeing in the old gauge.
            const auto identity=rtabmap::Transform::getIdentity();
            const auto *keyframe=map.getMemory()->getSignature(scanMapAnchor_);
            const bool secondCapture=chassisReset && keyframe && scanAnchor_ && !scanAnchor_->recovered &&
                !keyframe->getPose().isNull() && !scanAnchor_->pose.isNull() &&
                scanAnchor_->scan.stamp()>keyframe->getStamp() && data.stamp()>scanAnchor_->scan.stamp() &&
                data.stamp()-scanAnchor_->scan.stamp()<=maxLiveScanGap_;
            const auto supported=[&](const rtabmap::RegistrationInfo & info){
                return info.icpCorrespondences>=1000 && info.icpInliersRatio>=.20f &&
                    info.icpStructuralComplexity>=.02 && validCovariance(info.covariance);
            };
            const auto near=[&](const rtabmap::Transform & t,double meters,double degrees){
                return !t.isNull() && t.getNorm()<=meters && t.getAngle(identity)<=degrees*CV_PI/180.;
            };
            if(!secondCapture || !supported(result) || !near(relative,.05,3.)) return rtabmap::Transform();
            rtabmap::RegistrationInfo reverseInfo,secondInfo;
            const auto reverse=inertialIcp_->computeTransformation(data,reference,identity,&reverseInfo);
            if(reverse.isNull() || !supported(reverseInfo) || !near(relative*reverse,.01,1.))
                return rtabmap::Transform();
            const auto second=inertialIcp_->computeTransformation(scanAnchor_->scan,data,identity,&secondInfo);
            if(second.isNull() || !supported(secondInfo) || !near(relative.inverse()*keyframe->getPose().inverse()*scanAnchor_->pose*second,.02,1.))
                return rtabmap::Transform();
            return relative;
        };
        const auto identity=rtabmap::Transform::getIdentity();
        auto relative=rtabmap::Transform();
        if(scanMapGuess_)
        {
            const double age=data.stamp()-scanMapGuess_->stamp;
            if(age>0. && age<=2. && !scanMapGuess_->relative.isNull() &&
               scanMapGuess_->relative!=identity && cv::checkRange(scanMapGuess_->relative.dataMatrix()))
                relative=registerFrom(scanMapGuess_->relative);
        }
        if(relative.isNull()) relative=registerFrom(identity);
        int anchor=scanMapAnchor_;
        if(relative.isNull() && !sameGaugeGuess.isNull())
        {
            const auto match=matchEarlierScan(map,data,poses,sameGaugeGuess);
            if(match)
            {
                anchor=match->anchor;
                relative=match->relative;
                result=match->result;
            }
        }
        if(relative.isNull()) return false;
        const auto covariance=scanCovariance(result.covariance);
        cv::Mat information;
        if(!cv::invert(covariance,information,cv::DECOMP_CHOLESKY) || !cv::checkRange(information))
            return false;
        const auto location=poses.at(anchor)*relative;
        const auto previousPoses=poses;
        const auto links=map.getLocalConstraints();
        auto stats=statistics;
        stats["Motus/ScanRelocalizationOverlap/"]=result.icpInliersRatio;
        stats["Motus/ScanRelocalizationCorrespondences/"]=static_cast<float>(result.icpCorrespondences);
        beginLocalization(map);
        const rtabmap::Link measured(anchor,0,rtabmap::Link::kGlobalClosure,relative,information);
        if(!promote(map,data,measured,location,covariance,stats,poses,links,previousPoses,links))
            return false;
        scanMapAnchor_=anchor;
        scanMapGuess_=ScanMapGuess{relative,data.stamp()};
        return true;
    }

    void finishLocalization(rtabmap::Rtabmap & map, rtabmap::SensorData & data,
                            const rtabmap::Transform & external, const rtabmap::Transform & visual,
                            rtabmap::OdometryInfo & info, bool tracked, bool images,
                            const std::optional<ExternalRun> & externalRun)
    {
        const auto location=odom_;
        ++relocalizations_; state_=Localized;
        rememberScan(map,data);
        lastVisual_.setNull();
        if(tracked)
        {
            visualToNative_=location*visual.inverse();
            rememberVisualAlignment(data.stamp(),visual,info);
        }
        else if(images)
        {
            // This seed is a verified global pose, never frozen external odometry.
            estimator_->reset(location);
            rtabmap::OdometryInfo seed;
            const auto seeded=processVisual(data,seed);
            if(!seeded.isNull() && !seed.lost && estimator_->framesProcessed()==1)
                rememberVisualAlignment(data.stamp(),seeded,seed);
            visualToNative_=rtabmap::Transform::getIdentity();
        }
        aligned_=images;
        if(!images)
        {
            estimator_->reset(location);
            visualToNative_=rtabmap::Transform::getIdentity();
        }
        if(!external.isNull())
        {
            externalToNative_=location*external.inverse(); waiting_=false;
            // The closure independently aligns this measured run. Its next
            // pose can continue even if the local image tracker loses features.
            if(externalRun)
            {
                externalRun_=externalRun;
                lastAcceptedExternalRun_=externalRun;
                previousExternal_=external;
                previousExternalStamp_=data.stamp();
            }
        }
    }
    static constexpr double maxExternalScanGap_=6.;
    static constexpr double maxWitnessedGap_=5.5;
    static constexpr double maxLiveScanGap_=15.;
    static constexpr float minScanRelocalizationOverlap_=.30f;

    struct VisualAlignment
    {
        rtabmap::Transform visual,native;
        double stamp;
        unsigned int frames;
        std::map<int,cv::Point3f> landmarks;
    };

    void rememberVisualAlignment(double stamp,const rtabmap::Transform & visual,rtabmap::OdometryInfo & info)
    {
        visualLossStamp_=0.;
        if(info.localMap.size()>=static_cast<size_t>(minInliers_) && cv::checkRange(visual.dataMatrix()))
            visualAlignment_=VisualAlignment{visual,odom_,stamp,estimator_->framesProcessed(),std::move(info.localMap)};
        else
            visualAlignment_.reset();
    }

    static int countOldVisualInliers(const rtabmap::OdometryInfo & info,const std::map<int,cv::Point3f> & old)
    {
        std::set<int> inliers;
        for(int id:info.reg.inliersIDs)
        {
            const auto before=old.find(id), after=info.localMap.find(id);
            // IDs alone can recur after reset. Also check the metric landmark
            // in the F2M gauge, allowing small bundle-adjustment refinements.
            if(before!=old.end() && after!=info.localMap.end() && cv::norm(before->second-after->second)<=.02)
                inliers.insert(id);
        }
        return static_cast<int>(inliers.size());
    }

    bool hasVisualGeometry(const std::vector<cv::Point3f> & points) const
    {
        return std::count_if(points.begin(),points.end(),[](const cv::Point3f & p) {
            return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
        })>=minInliers_;
    }

    static cv::Mat takePayload(rtabmap::SensorData & data, const char * tag, int columns)
    {
        // Private SensorData records use RTAB-Map's compressed byte-matrix format.
        const auto & bytes=data.userDataCompressed();
        if(bytes.rows!=1 || bytes.type()!=CV_8UC1 || bytes.cols<=12 || bytes.cols>128)
            return {};
        int shape[3];
        std::memcpy(shape,bytes.data+bytes.cols-sizeof(shape),sizeof(shape));
        if(shape[0]!=1 || shape[1]!=columns || shape[2]!=CV_8UC1)
            return {};
        const auto payload=rtabmap::uncompressData(bytes);
        if(payload.empty() || std::memcmp(payload.data,tag,8)!=0)
            return {};
        data.setUserData(cv::Mat(),true);
        return payload;
    }

    static std::optional<ExternalRun> takeExternalRun(rtabmap::SensorData & data)
    {
        const auto payload=takePayload(data,"MOTUSOD1",24);
        if(payload.empty()) return {};
        ExternalRun token;
        std::memcpy(token.data(),payload.data+8,token.size());
        return token;
    }

    bool continuousMotion(const rtabmap::Transform & external, double stamp, double maxGap) const
    {
        const double dt=stamp-previousExternalStamp_;
        // The gateway witnesses the dense source between accepted native scans.
        if(external.isNull() || !(dt>0. && dt<=maxGap) || !cv::checkRange(external.dataMatrix()))
            return false;
        return plausibleMotion(previousExternal_.inverse()*external,dt);
    }

    static bool plausibleMotion(const rtabmap::Transform & delta,double dt)
    {
        return delta.getNorm()<=.15+1.5*dt &&
            delta.getAngle(rtabmap::Transform::getIdentity())<=10.*CV_PI/180.+2.5*dt;
    }

    static bool takePrediction(rtabmap::SensorData & data, rtabmap::Transform & guess)
    {
        const auto payload=takePayload(data,"MOTUSIMU",64);
        if(payload.empty()) return false;
        double p[7];
        std::memcpy(p,payload.data+8,sizeof(p));
        if(std::all_of(p,p+7,[](double v){return std::isfinite(v);}) &&
           std::abs(p[3]*p[3]+p[4]*p[4]+p[5]*p[5]+p[6]*p[6]-1.)<.002)
            guess=rtabmap::Transform(p[0],p[1],p[2],p[3],p[4],p[5],p[6]);
        return true;
    }

    struct ScanAnchor
    {
        rtabmap::SensorData scan;
        rtabmap::Transform pose;
        double referenceStamp; // Scan registrations do not extend their own recovery budget.
        bool recovered;
    };

    void rememberScan(rtabmap::Rtabmap & map, const rtabmap::SensorData & data)
    {
        if(data.laserScanRaw().empty()) {scanAnchor_.reset(); return;}
        const int node=map.getLastLocationId();
        if(state_==External || state_==VisualTracking || (state_==Localized && recoveryInliers_>0))
        {
            scanMapGuess_.reset();
            const auto *signature=map.getMemory()->getSignature(node);
            if(signature && map.getLocalOptimizedPoses().count(node))
            {
                scanMapAnchor_=node;
                const auto & anchorOdom=signature->getPose();
                if(!anchorOdom.isNull())
                    scanMapGuess_=ScanMapGuess{anchorOdom.inverse()*odom_,data.stamp()};
            }
        }
        rtabmap::SensorData scan;
        scan.setLaserScan(data.laserScanRaw());
        scan.setStamp(data.stamp());
        const double referenceStamp=state_==ScanTracking?scanAnchor_->referenceStamp:data.stamp();
        scanAnchor_=ScanAnchor{std::move(scan),odom_,referenceStamp,state_==Localized || state_==ScanTracking};
    }

    static cv::Mat scanCovariance(const cv::Mat & measurement)
    {
        cv::Mat covariance=measurement.clone();
        constexpr double positionVariance=.05*.05, angleVariance=(3.*CV_PI/180.)*(3.*CV_PI/180.);
        for(int i=0;i<6;++i)
            covariance.at<double>(i,i)=std::max(i<3?positionVariance:angleVariance,covariance.at<double>(i,i));
        return covariance;
    }

    bool registerScan(rtabmap::Rtabmap & map, rtabmap::SensorData & data,
                      const std::map<std::string,float> & statistics,double dt,bool waitingForMeasuredRun)
    {
        if(map.getLocalOptimizedPoses().empty()) return false;
        rtabmap::RegistrationInfo result;
        const auto relative=scanIcp_->computeTransformation(scanAnchor_->scan,data,
            rtabmap::Transform::getIdentity(),&result);
        // Require more than a small matching patch to replace a measured
        // stream temporarily or to bridge an explicitly scan-only origin.
        if(relative.isNull() || result.icpStructuralComplexity<.02 || result.icpCorrespondences<200 ||
           ((externalOrigin_ || waitingForMeasuredRun) && !(result.icpInliersRatio>=minScanRelocalizationOverlap_)) ||
           !validCovariance(result.covariance) || !plausibleMotion(relative,dt))
            return false;
        const auto measured=scanAnchor_->pose*relative;
        auto stats=statistics;
        stats["Motus/ScanTrackingOverlap/"]=result.icpInliersRatio;
        const auto covariance=scanCovariance(result.covariance);
        if(!map.process(data,measured,covariance,std::vector<float>(),stats))
            return false;
        // Carry only scan motion that the map accepted. Its map anchor stays fixed.
        if(scanMapGuess_ && scanMapGuess_->stamp==scanAnchor_->scan.stamp())
            scanMapGuess_=ScanMapGuess{scanMapGuess_->relative*relative,data.stamp()};
        state_=ScanTracking; odom_=measured;
        rememberScan(map,data); // Small motion may leave the persisted keyframe unchanged.
        return true;
    }

    std::optional<bool> registerPrediction(rtabmap::Rtabmap & map, rtabmap::SensorData & data,
                                          const rtabmap::Transform & guess,
                                          const std::map<std::string,float> & statistics)
    {
        state_=Searching; odom_.setNull(); visualInliers_=0;
        // A guess may extend only the already verified, active gauge.
        if(guess.isNull() || !cv::checkRange(guess.dataMatrix()) ||
           !initialized_ || waiting_ || missing_ || localizationMode_ ||
           map.getLocalOptimizedPoses().empty())
            return std::nullopt;
        const auto * last=map.getMemory()->getLastWorkingSignature();
        if(!last) return std::nullopt;
        rtabmap::RegistrationInfo result;
        const auto relative=inertialIcp_->computeTransformation(last->sensorData(),data,
            last->getPose().inverse()*externalToNative_*guess,&result);
        if(relative.isNull() || result.icpStructuralComplexity<.02 || !validCovariance(result.covariance))
            return std::nullopt; // Try independent recovery before consuming the input.
        const auto measured=last->getPose()*relative;
        auto stats=statistics;
        stats["Motus/InertialIcpOverlap/"]=result.icpInliersRatio;
        if(!map.process(data,measured,result.covariance,std::vector<float>(),stats))
            return false;
        state_=InertialRegistration; odom_=measured;
        return true;
    }

    rtabmap::Transform processVisual(rtabmap::SensorData & data, rtabmap::OdometryInfo & info)
    {
        struct RestoreScan
        {
            rtabmap::SensorData & data;
            rtabmap::LaserScan raw,compressed;
            ~RestoreScan() {data.setLaserScan(compressed); data.setLaserScan(raw,false);}
        } restore{data,data.laserScanRaw(),data.laserScanCompressed()};
        // Reg/Strategy=0 needs image geometry only. Keep its features on this
        // same envelope, while reserving scan processing for the native map.
        data.setLaserScan(rtabmap::LaserScan());
        return estimator_->process(data,&info);
    }
    static bool validCovariance(const cv::Mat & c)
    {
        if(c.type()!=CV_64FC1 || c.rows!=6 || c.cols!=6 || !cv::checkRange(c)) return false;
        for(int i=0;i<6;++i) if(c.at<double>(i,i)<=0) return false;
        return true;
    }
    WallAlignment walls_;
    std::unique_ptr<rtabmap::Odometry> estimator_;
    std::unique_ptr<rtabmap::RegistrationIcp> inertialIcp_,scanIcp_,revisitIcp_;
    rtabmap::RegistrationIcp neighborIcp_;
    int state_, minInliers_, maxOdomCacheSize_;
    float maxError_;
    bool missing_, waiting_, aligned_, localizationMode_;
    int relocalizations_, visualInliers_;
    bool imageMode_, initialized_;
    double visualLossStamp_=0.;
    std::optional<VisualAlignment> visualAlignment_;
    std::optional<ScanAnchor> scanAnchor_;
    int scanMapAnchor_=0;
    struct ScanMapGuess
    {
        rtabmap::Transform relative;
        double stamp;
    };
    std::optional<ScanMapGuess> scanMapGuess_;
    bool retryExternalNeighbor_=false;
    std::optional<ExternalRun> externalRun_, externalOrigin_, lastAcceptedExternalRun_;
    std::string proximityBeforeLocalization_;
    bool rgbdImageRecovery_;
    bool sawChassis_=false;
    rtabmap::Transform previousExternal_;
    double previousExternalStamp_=0.;
    double chassisOutageStart_=0.;
    double externalGraceStart_=0.;
    int recoveryNode_, recoveryAnchor_, recoveryInliers_;
    rtabmap::Transform externalToNative_, visualToNative_, odom_;
    rtabmap::Transform lastVisual_, lastNative_, initialVisualToNative_;
};
}

// Optional SensorData recovery using the pinned RTAB-Map algorithms.
// Coordinates: native_odom <- visual_odom / external_odom. No robot/frame policy.
#pragma once
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
public:
    enum State { External=0, Searching=1, Localized=2, VisualTracking=3, InertialRegistration=4 };
    explicit VisualRecovery(const rtabmap::ParametersMap & parameters):
        neighborIcp_(parameters), state_(External), missing_(false), waiting_(false), aligned_(false),
        localizationMode_(false), relocalizations_(0), visualInliers_(0), imageMode_(false), initialized_(false),
        recoveryNode_(0), recoveryAnchor_(0), recoveryInliers_(0)
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
            // Reject delayed observations before they can alter F2M. A clock
            // regression also invalidates the pending same-gauge recovery.
            visualLossStamp_=0.; visualAlignment_.reset(); visualInliers_=0;
            state_=Searching; odom_.setNull(); return false;
        }
        const auto externalRun=takeExternalRun(data);
        const double outageAge=data.stamp()-chassisOutageStart_;
        const bool recentOutage=chassisOutageStart_>0. && outageAge>=0. && outageAge<=2.;
        const bool continuousExternal=externalRun && externalRun==externalRun_ &&
            (!waiting_ || recentOutage) && !localizationMode_ && continuousMotion(external,data.stamp());
        const double externalAge=data.stamp()-previousExternalStamp_;
        const bool waitForChassis=external.isNull() && externalRun_ && !localizationMode_ &&
            externalAge>0. && externalAge<=maxExternalScanGap_ && (!chassisOutageStart_ || recentOutage) &&
            !map.getLocalOptimizedPoses().empty();
        if(waitForChassis && !chassisOutageStart_) chassisOutageStart_=data.stamp();
        if(!waitForChassis) chassisOutageStart_=0.;
        if(!continuousExternal && !waitForChassis) externalRun_.reset();
        rtabmap::Transform guess;
        if(external.isNull() && takePrediction(data,guess))
        {
            chassisOutageStart_=0.; externalRun_.reset();
            return registerPrediction(map,data,guess,statistics);
        }
        rtabmap::OdometryInfo info;
        rtabmap::Transform visual;
        const bool images=!data.imageRaw().empty() && !data.depthOrRightRaw().empty();
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
        if(!localizationMode_ && !continuousExternal && !waitForChassis && waiting_ && visualAlignment_ &&
           (visualLossStamp_>0. || (images && !tracked)))
        {
            const auto &anchor=*visualAlignment_;
            const double age=data.stamp()-anchor.stamp;
            const bool sameSequence=age>0. && age<=2. && framesBefore==anchor.frames &&
                estimator_->framesProcessed()==framesBefore+(tracked?1u:0u);
            aligned_=false;
            if(sameSequence && !tracked && external.isNull())
            {
                // Keep the old alignment as evidence only. The failed frame
                // must not create a pose, an edge, or enter localization mode.
                visualLossStamp_=data.stamp();
                state_=Searching; odom_.setNull(); return false;
            }
            visualLossStamp_=0.;
            if(sameSequence && tracked && cv::checkRange(visual.dataMatrix()) &&
               plausibleMotion(anchor.visual.inverse()*visual,age))
                reacquired=countOldVisualInliers(info,anchor.landmarks);
            if(reacquired>=minInliers_)
            {
                lastVisual_=anchor.visual; lastNative_=anchor.native;
                visualToNative_=lastNative_*lastVisual_.inverse();
                aligned_=true;
                externalGraceStart_=0.;
            }
            else
                visualAlignment_.reset();
        }
        std::map<std::string,float> reacquisitionStats;
        if(reacquired>=minInliers_)
        {
            reacquisitionStats=statistics;
            reacquisitionStats["Motus/VisualReacquiredInliers/"]=static_cast<float>(reacquired);
        }
        const auto &acceptedStats=reacquisitionStats.empty()?statistics:reacquisitionStats;
        if(waiting_ && aligned_ && !tracked && !continuousExternal)
        {
            if(!images)
            {
                state_=Searching; odom_.setNull();
                return false; // No observation reached VO; its established gauge is unchanged.
            }
            // A localization is not a motion estimate for a later observation.
            aligned_=false;
        }
        if(waiting_ && aligned_ && !external.isNull() && tracked)
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
        if(withinGrace && external.isNull() && (!images || info.features>0))
        {
            // A weak observation cannot move the map. Give the measured stream
            // time to return before RTAB-Map saves and reloads for localization.
            state_=Searching; odom_.setNull(); return false;
        }
        externalGraceStart_=0.;
        if(external.isNull())
            retryExternalNeighbor_=false;
        const bool verifiedRetry=waiting_ && !external.isNull() && (withinGrace || retryExternalNeighbor_) &&
            verifiesNeighbor(externalToNative_*external,withinGrace);
        if(verifiedRetry || continuousExternal)
            waiting_=false;
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
                if(accepted && tracked) rememberVisualAlignment(data.stamp(),visual,info);
                // A mature first keyframe establishes a gauge, but doesn't yet measure motion.
                // Only a later tracked frame may use this measured anchor during external loss.
                if(accepted && visualSeed) initialVisualToNative_=odom_*visual.inverse();
                if(accepted && externalRun)
                {
                    externalRun_=externalRun;
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
            if(accepted) rememberVisualAlignment(data.stamp(),visual,info);
            return accepted;
        }
        state_=Searching; odom_.setNull();
        // A short absence does not revoke the last independently witnessed
        // chassis run. Only a new measured pose carrying that same run may resume it.
        if(!images || waitForChassis) return false;
        retryExternalNeighbor_=false;
        if(!localizationMode_)
        {
            visualLossStamp_=0.;
            visualAlignment_.reset();
            // A missing odometry guess cannot justify proximity localization.
            // Native placeholder poses cannot constrain motion between queries.
            map.parseParameters({{"Mem/IncrementalMemory","false"}, {"RGBD/ProximityBySpace","false"},
                                 {"RGBD/MaxOdomCacheSize","0"}});
            localizationMode_=true;
        }
        // Failed localization can clear RTAB-Map's optimized cache when there
        // is no previous localization node. Keep the last accepted graph.
        const auto previousPoses=map.getLocalOptimizedPoses();
        const auto previousLinks=map.getLocalConstraints();
        const bool processed=map.process(data,rtabmap::Transform(),covariance,std::vector<float>(),statistics);
        const auto &stats=map.getStatistics();
        // Native loop id is emitted only after registration and graph validation.
        // Proximity and appearance hypotheses alone cannot authorize recovery.
        auto inliers=stats.data().find(rtabmap::Statistics::kLoopVisual_inliers());
        if(!processed || stats.loopClosureId()<=0 ||
           inliers==stats.data().end() || inliers->second<minInliers_ ||
           !validCovariance(stats.localizationCovariance()) || map.getLastLocalizationPose().isNull())
        {
            map.setOptimizedPoses(previousPoses,previousLinks);
            return processed;
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
            return processed;
        }

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
        restore["RGBD/ProximityBySpace"]="true";
        insertion["Mem/IncrementalMemory"]="true";
        insertion["RGBD/MaxOdomCacheSize"]=std::to_string(maxOdomCacheSize_);
        map.parseParameters(insertion); // Spatial proximity is already off while searching.
        localizationMode_=false;
        map.setOptimizedPoses(poses,links);
        odom_=location;
        const bool promoted=map.process(data,odom_,locCovariance,std::vector<float>(),statistics);
        map.parseParameters(restore);
        if(!promoted)
        {
            map.setOptimizedPoses(previousPoses,previousLinks);
            odom_.setNull(); aligned_=false; return false;
        }
        const int node=map.getLastLocationId();
        const auto *signature=map.getMemory()->getSignature(node);
        const bool inserted=signature && map.isInSTM(node) && !poses.count(node) &&
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
            map.setOptimizedPoses(previousPoses,previousLinks);
            odom_.setNull(); aligned_=false; return false;
        }
        recoveryNode_=node; recoveryAnchor_=anchor; recoveryInliers_=localizationInliers;
        ++relocalizations_; state_=Localized;
        lastVisual_.setNull();
        if(tracked)
        {
            visualToNative_=location*visual.inverse();
            rememberVisualAlignment(data.stamp(),visual,info);
        }
        else
        {
            // This seed is a verified global pose, never frozen external odometry.
            estimator_->reset(location);
            rtabmap::OdometryInfo seed;
            const auto seeded=processVisual(data,seed);
            if(!seeded.isNull() && !seed.lost && estimator_->framesProcessed()==1)
                rememberVisualAlignment(data.stamp(),seeded,seed);
            visualToNative_=rtabmap::Transform::getIdentity();
        }
        aligned_=true;
        if(!external.isNull())
        {
            externalToNative_=location*external.inverse(); waiting_=false;
        }
        return true;
    }
private:
    using ExternalRun=std::array<unsigned char,16>;
    static constexpr double maxExternalScanGap_=6.;

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

    bool continuousMotion(const rtabmap::Transform & external, double stamp) const
    {
        const double dt=stamp-previousExternalStamp_;
        // The gateway witnesses the dense source between accepted native scans.
        if(external.isNull() || !(dt>0. && dt<=maxExternalScanGap_) || !cv::checkRange(external.dataMatrix()))
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

    bool registerPrediction(rtabmap::Rtabmap & map, rtabmap::SensorData & data,
                            const rtabmap::Transform & guess,
                            const std::map<std::string,float> & statistics)
    {
        state_=Searching; odom_.setNull(); visualInliers_=0;
        // A guess may extend only the already verified, active gauge.
        if(guess.isNull() || !cv::checkRange(guess.dataMatrix()) ||
           !initialized_ || waiting_ || missing_ || localizationMode_ ||
           map.getLocalOptimizedPoses().empty())
            return false;
        const auto * last=map.getMemory()->getLastWorkingSignature();
        if(!last) return false;
        rtabmap::RegistrationInfo result;
        const auto relative=inertialIcp_->computeTransformation(last->sensorData(),data,
            last->getPose().inverse()*externalToNative_*guess,&result);
        if(relative.isNull() || result.icpStructuralComplexity<.02 || !validCovariance(result.covariance))
            return false;
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
    std::unique_ptr<rtabmap::Odometry> estimator_;
    std::unique_ptr<rtabmap::RegistrationIcp> inertialIcp_;
    rtabmap::RegistrationIcp neighborIcp_;
    int state_, minInliers_, maxOdomCacheSize_;
    float maxError_;
    bool missing_, waiting_, aligned_, localizationMode_;
    int relocalizations_, visualInliers_;
    bool imageMode_, initialized_;
    double visualLossStamp_=0.;
    std::optional<VisualAlignment> visualAlignment_;
    bool retryExternalNeighbor_=false;
    std::optional<ExternalRun> externalRun_;
    rtabmap::Transform previousExternal_;
    double previousExternalStamp_=0.;
    double chassisOutageStart_=0.;
    double externalGraceStart_=0.;
    int recoveryNode_, recoveryAnchor_, recoveryInliers_;
    rtabmap::Transform externalToNative_, visualToNative_, odom_;
    rtabmap::Transform lastVisual_, lastNative_, initialVisualToNative_;
};
}

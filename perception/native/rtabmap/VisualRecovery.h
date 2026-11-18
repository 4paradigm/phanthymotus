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
#include <rtabmap/utilite/ULogger.h>
#include <memory>
#include <algorithm>
#include <cmath>

namespace motus {
class VisualRecovery
{
public:
    enum State { External=0, Searching=1, Localized=2, VisualTracking=3 };
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
        if(images) visual=processVisual(data,info);
        visualInliers_=info.reg.inliers;
        const bool tracked=!visual.isNull() && !info.lost &&
            visualInliers_>=minInliers_ && validCovariance(info.reg.covariance);
        if(images && !tracked)
        {
            lastVisual_.setNull(); lastNative_.setNull();
        }
        if(external.isNull() && !missing_)
        {
            waiting_=true; aligned_=!lastVisual_.isNull();
            if(aligned_) visualToNative_=lastNative_*lastVisual_.inverse();
        }
        missing_=external.isNull();
        if(waiting_ && aligned_ && !tracked)
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
        if(!waiting_ && !external.isNull())
        {
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
            if(!tracked || lastVisual_.isNull())
            {
                // Validate the supplied constraint; never replace it with an ICP pose.
                // Memory::computeIcpTransform uses the stricter merged-scan overlap.
                const auto *last=map.getMemory()->getLastWorkingSignature();
                if(last && neighborIcp_.computeTransformation(last->sensorData(),data,
                    last->getPose().inverse()*odom_).isNull())
                {
                    waiting_=true; aligned_=!lastVisual_.isNull();
                    if(aligned_) visualToNative_=lastNative_*lastVisual_.inverse();
                    state_=Searching; odom_.setNull(); return false;
                }
            }
            if(tracked)
            {
                lastVisual_=visual; lastNative_=odom_;
            }
            return map.process(data,odom_,covariance,velocity,statistics);
        }
        if(aligned_ && tracked)
        {
            state_=VisualTracking; odom_=visualToNative_*visual;
            lastVisual_=visual; lastNative_=odom_;
            return map.process(data,odom_,info.reg.covariance,std::vector<float>(),statistics);
        }
        state_=Searching; odom_.setNull();
        if(!images) return false;
        if(!localizationMode_)
        {
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
            visualToNative_=location*visual.inverse();
        else
        {
            // This seed is a verified global pose, never frozen external odometry.
            estimator_->reset(location);
            rtabmap::OdometryInfo seed;
            processVisual(data,seed);
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
    rtabmap::RegistrationIcp neighborIcp_;
    int state_, minInliers_, maxOdomCacheSize_;
    float maxError_;
    bool missing_, waiting_, aligned_, localizationMode_;
    int relocalizations_, visualInliers_;
    bool imageMode_, initialized_;
    int recoveryNode_, recoveryAnchor_, recoveryInliers_;
    rtabmap::Transform externalToNative_, visualToNative_, odom_;
    rtabmap::Transform lastVisual_, lastNative_;
};
}

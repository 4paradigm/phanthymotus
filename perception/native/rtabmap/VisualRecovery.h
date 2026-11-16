// Optional SensorData recovery using the pinned RTAB-Map algorithms.
// Coordinates: native_odom <- visual_odom / external_odom. No robot/frame policy.
#pragma once
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/Memory.h>
#include <rtabmap/core/Signature.h>
#include <rtabmap/core/Odometry.h>
#include <rtabmap/core/OdometryInfo.h>
#include <memory>
#include <algorithm>
#include <cmath>

namespace motus {
class VisualRecovery
{
public:
    enum State { External=0, Searching=1, Localized=2, VisualTracking=3 };
    explicit VisualRecovery(const rtabmap::ParametersMap & parameters):
        state_(External), missing_(false), waiting_(false), aligned_(false),
        localizationMode_(false), relocalizations_(0), visualInliers_(0), imageMode_(false), initialized_(false)
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
    }
    int state() const {return state_;}
    int relocalizations() const {return relocalizations_;}
    int visualInliers() const {return visualInliers_;}
    const rtabmap::Transform & odom() const {return odom_;}

    static void finalize(rtabmap::Rtabmap & map)
    {
        std::map<int,rtabmap::Transform> poses;
        std::multimap<int,rtabmap::Link> links;
        map.getGraph(poses,links,true,true); // Native global optimization, no sensor blobs.
        const auto &local=map.getLocalOptimizedPoses();
        if(!poses.empty() && (local.empty() || poses.count(local.begin()->first)))
            map.setOptimizedPoses(poses,links);
    }

    bool process(rtabmap::Rtabmap & map, rtabmap::SensorData & data,
                 const rtabmap::Transform & external, const cv::Mat & covariance,
                 const std::vector<float> & velocity=std::vector<float>(),
                 const std::map<std::string,float> & statistics=std::map<std::string,float>())
    {
        rtabmap::OdometryInfo info;
        rtabmap::Transform visual;
        const bool images=!data.imageRaw().empty() && !data.depthOrRightRaw().empty();
        if(!initialized_)
        {
            waiting_=!map.getLocalOptimizedPoses().empty();
            initialized_=true;
        }
        if(images!=imageMode_)
        {
            // Keep the supplied motion covariance. Scan ICP on a weakly
            // constrained surface can replace it with an overconfident edge.
            // Visual candidates still require native visual geometry AND ICP.
            map.parseParameters({{"RGBD/NeighborLinkRefining",images?"false":"true"},
                                 {"RGBD/LoopCovLimited",images?"true":"false"},
                                 {"RGBD/ProximityOdomGuess",images?"false":"true"}});
            imageMode_=images;
        }
        if(images) visual=estimator_->process(data, &info);
        visualInliers_=info.reg.inliers;
        const bool tracked=!visual.isNull() && !info.lost &&
            visualInliers_>=minInliers_ && validCovariance(info.reg.covariance);
        if(external.isNull() && !missing_)
        {
            waiting_=true; aligned_=false;
        }
        missing_=external.isNull();
        if(waiting_ && aligned_ && !tracked)
        {
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
        if(!localizationMode_)
        {
            // A missing odometry guess cannot justify proximity localization.
            map.parseParameters({{"Mem/IncrementalMemory","false"}, {"RGBD/ProximityBySpace","false"}});
            localizationMode_=true;
        }
        const bool processed=map.process(data,rtabmap::Transform(),covariance,std::vector<float>(),statistics);
        const auto &stats=map.getStatistics();
        // Native loop id is emitted only after registration and graph validation.
        // Proximity and appearance hypotheses alone cannot authorize recovery.
        auto inliers=stats.data().find(rtabmap::Statistics::kLoopVisual_inliers());
        if(!processed || !images || stats.loopClosureId()<=0 ||
           inliers==stats.data().end() || inliers->second<minInliers_ ||
           !validCovariance(stats.localizationCovariance()) || map.getLastLocalizationPose().isNull())
            return processed;

        const int anchor=stats.loopClosureId(), query=stats.refImageId();
        const auto location=map.getLastLocalizationPose();
        const auto anchorPose=map.getPose(anchor);
        const auto locCovariance=stats.localizationCovariance().clone();
        auto poses=map.getLocalOptimizedPoses();
        auto links=map.getLocalConstraints();
        poses.erase(query);
        for(auto i=links.begin();i!=links.end();)
            if(i->second.from()==query || i->second.to()==query) i=links.erase(i); else ++i;
        if(anchorPose.isNull() || poses.find(anchor)==poses.end()) return processed;

        // Pinned parseParameters clears optimized poses when returning to mapping.
        // Preserve them explicitly, and promote this SAME geometrically verified
        // observation with a checked native edge into the original graph.
        map.parseParameters({{"Mem/IncrementalMemory","true"}, {"RGBD/ProximityBySpace","true"}});
        localizationMode_=false;
        map.setOptimizedPoses(poses,links);
        odom_=location;
        if(!map.process(data,odom_,locCovariance,std::vector<float>(),statistics))
        {
            map.setOptimizedPoses(poses,links);
            aligned_=false; return false;
        }
        const int node=map.getLastLocationId();
        const auto *signature=map.getMemory()->getSignature(node);
        bool connected=signature && signature->hasLink(anchor);
        if(!connected)
            connected=map.addLink(rtabmap::Link(anchor,node,rtabmap::Link::kGlobalClosure,
                anchorPose.inverse()*location,map.getInformation(locCovariance)));
        if(!connected)
        {
            map.deleteLastLocation(); map.setOptimizedPoses(poses,links);
            odom_.setNull(); aligned_=false; return false;
        }
        ++relocalizations_; state_=Localized;
        lastVisual_.setNull();
        if(tracked)
            visualToNative_=location*visual.inverse();
        else
        {
            // This seed is a verified global pose, never frozen external odometry.
            estimator_->reset(location);
            rtabmap::OdometryInfo seed;
            estimator_->process(data,&seed);
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
    static bool validCovariance(const cv::Mat & c)
    {
        if(c.type()!=CV_64FC1 || c.rows!=6 || c.cols!=6 || !cv::checkRange(c)) return false;
        for(int i=0;i<6;++i) if(c.at<double>(i,i)<=0) return false;
        return true;
    }
    std::unique_ptr<rtabmap::Odometry> estimator_;
    int state_, minInliers_;
    float maxError_;
    bool missing_, waiting_, aligned_, localizationMode_;
    int relocalizations_, visualInliers_;
    bool imageMode_, initialized_;
    rtabmap::Transform externalToNative_, visualToNative_, odom_;
    rtabmap::Transform lastVisual_, lastNative_;
};
}

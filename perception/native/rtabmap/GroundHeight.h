// Vertical odometry drift measured from repeated observations of the same ground cells.
#pragma once
#include <rtabmap/core/LocalGridMaker.h>
#include <rtabmap/core/Signature.h>
#include <opencv2/flann.hpp>
#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <optional>
#include <vector>

namespace motus {
namespace ground_height {
using XYZ=cv::Vec3d;
using Cell=std::pair<int,int>;
using Cells=std::map<Cell,double>;
inline double median(std::vector<double> values) {
    std::sort(values.begin(),values.end());
    return .5*(values[values.size()/2]+values[(values.size()-1)/2]);
}
inline std::vector<XYZ> points(const cv::Mat &cells,const rtabmap::Transform &pose) {
    std::vector<XYZ> out;
    if(cells.rows!=1 || cells.depth()!=CV_32F || cells.channels()<3) return out;
    for(int c=0;c<cells.cols;++c) {
        const float *p=cells.ptr<float>()+c*cells.channels();
        XYZ q;
        for(int k=0;k<3;++k) {
            const float *m=pose.data()+4*k;
            q[k]=m[0]*p[0]+m[1]*p[1]+m[2]*p[2]+m[3];
        }
        if(std::isfinite(q[0]) && std::isfinite(q[1]) && std::isfinite(q[2])) out.push_back(q);
    }
    return out;
}
inline bool confirms(const std::vector<XYZ> &target,const std::vector<XYZ> &src,double dz) {
    if(target.size()<40 || src.size()<40) return false;
    cv::Mat ref,query;
    cv::Mat(int(target.size()),3,CV_64F,const_cast<XYZ*>(target.data())).convertTo(ref,CV_32F);
    cv::Mat(int(src.size()),3,CV_64F,const_cast<XYZ*>(src.data())).convertTo(query,CV_32F);
    // The randomized forest consumes the thread's RNG used by visual registration.
    cv::flann::IndexParams tree;
    tree.setAlgorithm(cvflann::FLANN_INDEX_KDTREE_SINGLE);
    cv::flann::Index index(ref,tree);
    cv::Mat ids,before,after;
    index.knnSearch(query,ids,before,1,cv::flann::SearchParams(-1));
    query.col(2)+=dz;
    index.knnSearch(query,ids,after,1,cv::flann::SearchParams(-1));
    int count=0; double improvement=0.;
    for(int i=0;i<query.rows;++i) {
        double a=std::sqrt(before.at<float>(i)), b=std::sqrt(after.at<float>(i));
        if(std::min(a,b)>=.25) continue;
        improvement+=std::min(a,.25)-std::min(b,.25);++count;
    }
    return count>=40 && improvement>.002*count;
}
inline std::optional<double> match(const Cells &a,const Cells &b) {
    std::vector<double> dz[2]; std::vector<cv::Vec2d> xy[2];
    for(const auto &kv:a) {
        auto hit=b.find(kv.first); if(hit==b.end()) continue;
        const auto [x,y]=kv.first; int group=(x+y)&1;
        dz[group].push_back(kv.second-hit->second); xy[group].emplace_back(.18*x,.18*y);
    }
    double center[2];
    for(int g=0;g<2;++g) {
        if(dz[g].size()<12) return {};
        cv::Mat samples(int(xy[g].size()),2,CV_64F,xy[g].data());
        cv::PCA pca(samples,cv::Mat(),cv::PCA::DATA_AS_ROW);
        if(pca.eigenvalues.at<double>(1)<.02) return {};
        center[g]=median(dz[g]);
        for(auto &v:dz[g]) v=std::abs(v-center[g]);
        if(median(dz[g])>.025) return {};
    }
    if(std::abs(center[0]-center[1])>.04) return {};
    return .5*(center[0]+center[1]);
}
}

class GroundHeight
{
    using Cells=ground_height::Cells;
    using Points=std::vector<ground_height::XYZ>;
    struct Sample {double stamp; Cells cells; Points obstacles;};
    std::deque<Sample> history_;
    rtabmap::LocalGridMaker grid_;
    double offset_=0.;
public:
    explicit GroundHeight(const rtabmap::ParametersMap &params): grid_(params) {}
    double offset() const {return offset_;}
    void clearObservations() {history_.clear();}

    void observe(const rtabmap::SensorData &data,const rtabmap::Transform &pose)
    {
        // Project only measured cells, including the scan's capture-time sensor mount.
        cv::Mat floor,obstacles,empty;
        cv::Point3f viewpoint;
        rtabmap::Signature signature(0,0,0,data.stamp(),"",pose,rtabmap::Transform(),data);
        grid_.createLocalMap(signature,floor,obstacles,empty,viewpoint);
        rtabmap::Transform projected(pose.dataMatrix().clone());
        projected.data()[11]+=offset_;
        std::map<ground_height::Cell,std::vector<double>> buckets;
        for(const auto &p:ground_height::points(floor,projected))
            buckets[{int(std::floor(p[0]/.18)),int(std::floor(p[1]/.18))}].push_back(p[2]);
        Sample now{data.stamp(),{},ground_height::points(obstacles,projected)};
        for(const auto &bucket:buckets)
        {
            const auto &values=bucket.second;
            const auto range=std::minmax_element(values.begin(),values.end());
            if(values.size()>=3 && *range.second-*range.first<.08)
                now.cells.emplace(bucket.first,ground_height::median(values));
        }
        while(!history_.empty() && now.stamp-history_.front().stamp>2.5)
            history_.pop_front();
        if(now.cells.empty()) return;
        std::vector<double> proposals{0.};
        for(const auto &previous:history_)
        {
            const auto dz=ground_height::match(previous.cells,now.cells);
            if(!dz) continue;
            // Larger corrections must independently improve nonground alignment both ways.
            if(std::abs(*dz)>.05 &&
               (!ground_height::confirms(previous.obstacles,now.obstacles,*dz) ||
                !ground_height::confirms(now.obstacles,previous.obstacles,-*dz))) continue;
            proposals.push_back(*dz);
        }
        const double dz=ground_height::median(proposals);
        for(auto &cell:now.cells) cell.second+=dz;
        for(auto &p:now.obstacles) p[2]+=dz;
        if(history_.size()==32) history_.pop_front();
        history_.push_back(std::move(now));
        offset_+=dz;
    }
};
}

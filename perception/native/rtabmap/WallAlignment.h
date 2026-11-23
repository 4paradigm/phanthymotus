// Opportunistic wall constraints for retained, continuously tracked keyframes.
#pragma once
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/Memory.h>
#include <rtabmap/core/Signature.h>
#include <opencv2/imgproc.hpp>
#include <deque>
#include <optional>

namespace motus {
class WallAlignment
{
    using Vec=cv::Vec3f;
    using Pose=rtabmap::Transform;
    static Vec rotate(const Pose & t,const Vec & v)
    {
        return {t.r11()*v[0]+t.r12()*v[1]+t.r13()*v[2],
                t.r21()*v[0]+t.r22()*v[1]+t.r23()*v[2],
                t.r31()*v[0]+t.r32()*v[1]+t.r33()*v[2]};
    }
    static Vec position(const Pose & t) {return {t.x(),t.y(),t.z()};}
    static Vec apply(const Pose & t,const Vec & v) {return rotate(t,v)+position(t);}
    struct Patch
    {
        int node;
        double stamp;
        Vec first, last; // Endpoints in the node frame.
        std::vector<Vec> context; // Independent returns away from the fitted wall.
    };
    std::deque<Patch> history_;
    double lastSample_=0.;

    static std::optional<Patch> extract(int node,double stamp,const rtabmap::LaserScan & scan,const Pose & pose)
    {
        if(scan.empty() || scan.is2d() || scan.channels()<3 || scan.data().depth()!=CV_32F ||
           !scan.data().isContinuous()) return {};
        const Pose sensor=pose*scan.localTransform();
        std::vector<Vec> points;
        std::map<std::pair<int,int>,std::pair<float,float>> columns;
        const int stride=std::max(1,(scan.size()+2047)/2048);
        for(int i=0;i<scan.size();i+=stride)
        {
            const float *xyz=scan.data().ptr<float>()+i*scan.channels();
            const Vec p=apply(sensor,Vec(xyz));
            if(!(cv::norm(p-position(sensor))<=8.f)) continue;
            points.push_back(p);
            auto key=std::make_pair(int(std::floor(p[0]/.15f)),int(std::floor(p[1]/.15f)));
            auto & limits=columns.emplace(key,std::make_pair(p[2],p[2])).first->second;
            limits.first=std::min(limits.first,p[2]); limits.second=std::max(limits.second,p[2]);
        }
        std::vector<cv::Point2f> tall;
        for(const auto & c:columns)
            if(c.second.second-c.second.first>.65f)
                tall.emplace_back((c.first.first+.5f)*.15f,(c.first.second+.5f)*.15f);
        if(tall.size()<9) return {};
        // Bounded deterministic hypotheses, each tall column gets one vote.
        std::vector<cv::Point2f> best;
        for(size_t a=0;a<tall.size();a+=std::max<size_t>(1,tall.size()/12))
            for(size_t b=a+1;b<tall.size();b+=std::max<size_t>(1,tall.size()/12))
            {
                const cv::Point2f delta=tall[b]-tall[a]; const double length=cv::norm(delta);
                if(length<.8) continue;
                std::vector<cv::Point2f> support;
                for(const auto & q:tall)
                    if(std::abs(delta.cross(q-tall[a]))<.10*length) support.push_back(q);
                if(support.size()>best.size()) best=std::move(support);
            }
        if(best.size()<9) return {};
        cv::Vec4f line;
        cv::fitLine(best,line,cv::DIST_L2,0,.01,.01);
        Vec center(line[2],line[3],sensor.z()), tangent(line[0],line[1],0);
        Vec normal(-line[1],line[0],0);
        float lo=0,hi=0;
        for(const auto & q:best)
        {
            const float t=(q.x-center[0])*line[0]+(q.y-center[1])*line[1];
            lo=std::min(lo,t); hi=std::max(hi,t);
        }
        if(hi-lo<1.2f) return {};
        const Pose inverse=pose.inverse();
        Patch patch{node,stamp,apply(inverse,center+lo*tangent),apply(inverse,center+hi*tangent),{}};
        // Keep context above the floor and off the wall. A parallel bare wall
        // alone cannot independently distinguish the two sides of a corridor.
        for(size_t i=1;i<points.size();i+=std::max<size_t>(1,points.size()/256))
        {
            const auto & p=points[i];
            if(p[2]>pose.z()+.22f && std::abs(normal.dot(p-center))>.18f)
                patch.context.push_back(apply(inverse,p));
        }
        return patch;
    }

public:
    void reset() {history_.clear();lastSample_=0.;}

    void observe(rtabmap::Rtabmap & map,const rtabmap::SensorData & data)
    {
        const int node=map.getLastLocationId();
        const auto & poses=map.getLocalOptimizedPoses();
        const auto *signature=map.getMemory()->getSignature(node);
        if(!signature || signature->getStamp()!=data.stamp() || !poses.count(node) ||
           data.stamp()-lastSample_<1.) return;
        lastSample_=data.stamp();
        const Pose currentPose=poses.at(node);
        auto current=extract(node,data.stamp(),data.laserScanRaw(),currentPose);
        if(!current) return;
        const Vec a=apply(currentPose,current->first),b=apply(currentPose,current->last);
        Vec dir=b-a; dir[2]=0; dir=cv::normalize(dir);
        const Vec center=(a+b)*.5f;
        const Patch *best=nullptr;
        float bestScore=0;
        Pose bestPose;
        Vec bestNormal;
        for(const auto & old:history_)
        {
            if(current->stamp-old.stamp<10. || !poses.count(old.node) || signature->hasLink(old.node)) continue;
            const Pose previous=poses.at(old.node);
            const Vec c=apply(previous,old.first),d=apply(previous,old.last);
            Vec axis=d-c; axis[2]=0;
            const float length=cv::norm(axis);
            if(length<1.f) continue;
            axis/=length;
            const Vec normal(-axis[1],axis[0],0);
            const float start=(a-c).dot(axis),end=(b-c).dot(axis);
            const float overlap=std::min(length,std::max(start,end))-std::max(0.f,std::min(start,end));
            const float offset=(c-center).dot(normal);
            if(overlap<.85f || std::abs(axis.dot(dir))<.99f || std::abs(offset)>.37f ||
               normal.dot(position(currentPose)-c)*normal.dot(position(previous)-c)<=0) continue;
            if(axis.dot(dir)<0) axis=-axis;
            const float yaw=std::atan2(dir.cross(axis)[2],dir.dot(axis));
            const Pose rotation(0,0,0,0,0,yaw);
            const Vec shift=center+offset*normal-rotate(rotation,center);
            const Pose fix(shift[0],shift[1],shift[2],0,0,yaw);
            // Score the correction on independent 3-D structure observed by both
            // scans; a better fit to wall returns alone is insufficient.
            const Pose before=previous.inverse()*currentPose, after=previous.inverse()*fix*currentPose;
            if(current->context.size()<12 || old.context.size()<12) continue;
            int matches=0; float gain=0;
            for(const auto & p:current->context)
            {
                const Vec x=apply(before,p),y=apply(after,p);
                float pre=.35f*.35f,post=pre;
                for(const auto & ref:old.context)
                {
                    pre=std::min(pre,(ref-x).dot(ref-x));
                    post=std::min(post,(ref-y).dot(ref-y));
                }
                if(post<.18f*.18f) ++matches;
                gain+=std::sqrt(pre)-std::sqrt(post);
            }
            const float score=gain/current->context.size();
            if(matches>=12 && matches>=.2f*current->context.size() && score>.006f && score>bestScore)
            {
                best=&old;bestScore=score;bestPose=after;bestNormal=rotate(previous.inverse(),normal);
            }
        }
        if(best && map.getMemory()->getSignature(best->node))
        {
            // Normal displacement and yaw are observed. Retain only numerical
            // regularization along the wall and in the other rotational axes.
            cv::Mat info=cv::Mat::eye(6,6,CV_64F)*.01;
            for(int i=0;i<3;++i) for(int j=0;j<3;++j)
                info.at<double>(i,j)+=400.*bestNormal[i]*bestNormal[j];
            const Vec up=rotate(poses.at(best->node).inverse(),Vec(0,0,1));
            for(int i=0;i<3;++i) for(int j=0;j<3;++j)
                info.at<double>(3+i,3+j)+=1600.*up[i]*up[j];
            map.addLink(rtabmap::Link(best->node,node,rtabmap::Link::kLocalSpaceClosure,
                        bestPose,info));
        }
        history_.push_back(std::move(*current));
        if(history_.size()>96) history_.pop_front();
    }
};
}

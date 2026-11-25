// Run with the path of an empty directory for the three native databases.
#include "WallAlignment.h"
#include <stdexcept>

static void replay(const std::string & database,bool differentScene,bool reset)
{
    rtabmap::ParametersMap parameters{{"Reg/Strategy","1"},{"RGBD/NeighborLinkRefining","false"},
        {"RGBD/ProximityBySpace","false"},{"RGBD/ProximityByTime","false"},{"Grid/Sensor","0"},
        {"Mem/BadSignaturesIgnored","false"},{"RGBD/LinearUpdate","0.01"},{"Optimizer/Strategy","0"}};
    rtabmap::Rtabmap map;
    map.init(parameters,database);
    motus::WallAlignment walls;
    for(int i=0;i<3;++i)
    {
        const float y=i*.2f, drift=i==2?.15f:0.f;
        std::vector<cv::Vec3f> points;
        for(int k=0;k<32;++k) for(int h=0;h<16;++h)
            points.emplace_back(2.f,k*.1f-1.5f-y,h*.10f);
        // A repeated wall with unrelated context at a different height must
        // produce no edge even when the odometry places the two walls close.
        for(int k=0;k<20;++k) for(int j=0;j<10;++j)
            points.emplace_back(.9f+.07f*std::sin(k*.4f),k*.06f-y,.35f+j*.025f+
                                (differentScene && i==2?1.4f:0.f));
        const cv::Mat xyz(1,int(points.size()),CV_32FC3,points.data());
        rtabmap::SensorData data;
        data.setStamp(i==2?13.:1.+i*3);
        data.setLaserScan(rtabmap::LaserScan(xyz.clone(),0,0.f,rtabmap::LaserScan::kXYZ));
        if(!map.process(data,rtabmap::Transform(drift,y,0,0,0,0),cv::Mat::eye(6,6,CV_64F)*.0025))
            throw std::runtime_error("native fixture failed to admit measured pose");
        if(reset && i==2) walls.reset();
        walls.observe(map,data);
    }
    map.close();
    map.init(parameters,database);
    const auto *last=map.getMemory()->getSignature(3);
    if(map.getLocalOptimizedPoses().size()!=3 || !last ||
       last->hasLink(1,rtabmap::Link::kLocalSpaceClosure)!=(!differentScene && !reset))
        throw std::runtime_error("incorrect persisted wall constraint in "+database);
    if(!differentScene && !reset)
    {
        const auto info=last->getLinks().find(1)->second.infMatrix();
        const auto & odom=map.getMemory()->getOdomMaxInf();
        for(int i=0;i<6;++i)
            if(info.at<double>(i,i)>odom.at(i)+1e-5)
                throw std::runtime_error("wall constraint exceeded measured neighbor information");
    }
    map.close(false);
}

int main(int argc,char **argv)
{
    if(argc!=2) return 2;
    const std::string root=argv[1];
    replay(root+"/same.db",false,false);
    replay(root+"/different.db",true,false);
    replay(root+"/reset.db",false,true);
}

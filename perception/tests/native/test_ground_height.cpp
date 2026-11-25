// A static sloping floor viewed from a body that can move vertically.
#include "GroundHeight.h"
#include <rtabmap/core/LaserScan.h>
#include <iostream>
#include <stdexcept>

int main()
{
    using namespace rtabmap;
    ParametersMap params{{"Grid/Sensor","0"},{"Grid/RangeMax","8"},
                         {"Grid/3D","true"},{"Grid/CellSize","0.04"},
                         {"Grid/NormalsSegmentation","false"},{"Grid/MaxGroundHeight","0.35"}};
    motus::GroundHeight tracker(params);
    auto observe=[&](double stamp, float x, float odomZ, float trueZ)
    {
        cv::Mat samples(1,80*80,CV_32FC3);
        for(int i=0;i<80;++i)
            for(int j=0;j<80;++j)
            {
                const float ahead=.8f+.032f*i, left=-1.28f+.032f*j;
                samples.at<cv::Vec3f>(0,80*i+j)={ahead,left,.03f*(ahead+x)-trueZ};
            }
        SensorData data;
        data.setStamp(stamp);
        data.setLaserScan(LaserScan(samples,0,0,LaserScan::kXYZ));
        Transform pose(x,0.f,odomZ,0.f,0.f,0.f);
        tracker.observe(data,pose);
        if(pose.z()!=odomZ) throw std::runtime_error("mutated source pose");
        return tracker.offset();
    };
    if(observe(1.,0.f,0.f,0.f)!=0.) return 1;
    if(std::abs(observe(1.4,.12f,.08f,.08f))>.01)
        throw std::runtime_error("real 8 cm lift over a sloping floor was suppressed");
    const double drift=observe(1.8,.18f,.04f,0.f);
    if(drift>-.01 || drift<-.05) throw std::runtime_error("overlapping surfaces did not correct odometry drift");
    if(observe(2.,8.f,.10f,0.f)!=drift)
        throw std::runtime_error("disjoint surfaces changed previous height correction");
    if(observe(5.,0.f,.13f,0.f)!=drift)
        throw std::runtime_error("expired surfaces affected a later revisit");
    tracker.clearObservations();
    if(observe(5.3,0.f,.04f,0.f)!=drift)
        throw std::runtime_error("a gap reused old surfaces or lost the confirmed gauge offset");
    std::cout<<"PASS observed height drift correction "<<drift<<" m"<<std::endl;
}

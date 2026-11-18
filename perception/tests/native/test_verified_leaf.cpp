// Pass a disposable map copy, its original, and the ID of an archived RGB-D revisit.
#include <rtabmap/core/DBDriverSqlite3.h>
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/Memory.h>
#include <rtabmap/core/Signature.h>
#include <algorithm>
#include <filesystem>
#include <iterator>
#include <stdexcept>

using namespace rtabmap;

int main(int argc, char ** argv)
{
    if(argc!=4 || !std::filesystem::is_regular_file(argv[1]) || !std::filesystem::is_regular_file(argv[2]) ||
       std::filesystem::equivalent(argv[1],argv[2]))
        throw std::runtime_error("expected distinct writable map copy, source database and revisit node ID");
    DBDriverSqlite3 source;
    if(!source.openConnection(argv[2])) throw std::runtime_error("cannot open source database");
    SensorData query;
    const int sourceId=std::stoi(argv[3]);
    source.getNodeData(sourceId,query,true,true,false,false);
    Transform original,truth; int mapId,sourceWeight; std::string label; double stamp;
    std::vector<float> velocity; GPS gps; EnvSensors sensors;
    if(!source.getNodeInfo(sourceId,original,mapId,sourceWeight,label,stamp,truth,velocity,gps,sensors))
        throw std::runtime_error("cannot read the revisit stamp");
    query.setStamp(stamp);
    query.uncompressData();
    auto params=source.getLastParameters();
    source.closeConnection(false);
    if(query.imageRaw().empty() || query.depthOrRightRaw().empty() || query.laserScanRaw().isEmpty())
        throw std::runtime_error("the revisit needs recorded RGB, depth and scan");

    Rtabmap map;
    map.init(params,argv[1]);
    map.parseParameters({{"Mem/IncrementalMemory","false"},{"RGBD/ProximityBySpace","false"},
                         {"RGBD/MaxOdomCacheSize","0"}});
    const auto history=map.getLocalOptimizedPoses();
    const auto constraints=map.getLocalConstraints();
    query.setId(0);
    if(!map.process(query,Transform(),cv::Mat::eye(6,6,CV_64F)))
        throw std::runtime_error("native localization did not process the query");
    const auto & stats=map.getStatistics();
    const int anchor=stats.loopClosureId();
    const auto & queryEdges=stats.getSignaturesData().at(stats.refImageId()).getLinks();
    const auto measured=queryEdges.find(anchor);
    const auto location=map.getLastLocalizationPose();
    const auto covariance=stats.localizationCovariance().clone();
    if(!history.count(anchor) || measured==queryEdges.end() || measured->second.type()!=Link::kGlobalClosure ||
       location.isNull() || covariance.empty())
        throw std::runtime_error("test requires a genuine native localization against the original map");
    const Link registration=measured->second.inverse();

    // Isolate attachment from appearance retrieval. The edge above was measured
    // from the full observation. Its scan opens a new mapping session at that pose.
    map.parseParameters({{"Mem/IncrementalMemory","true"}});
    map.setOptimizedPoses(history,constraints);
    SensorData scan;
    scan.setLaserScan(query.laserScanRaw());
    scan.setStamp(query.stamp());
    if(!map.process(scan,location,covariance)) throw std::runtime_error("cannot insert isolated scan");
    const int node=map.getLastLocationId();
    const auto * tip=map.getMemory()->getSignature(node);
    if(!tip || !tip->getLinks().empty() || history.count(node) || history.size()<3)
        throw std::runtime_error("test requires an isolated new tip and connected history");
    const Link edge(anchor,node,Link::kGlobalClosure,registration.transform(),registration.infMatrix());
    const auto poses=map.getLocalOptimizedPoses();
    const auto links=map.getLocalConstraints();
    const int weight=tip->getWeight();

    for(const auto & fault : {Transform(2.f,0.f,0.f,0.f,0.f,0.f),Transform(0.f,0.f,0.f,0.f,.8f,0.f)})
    {
        auto broken=poses;
        const int old=std::next(history.begin())->first;
        broken.at(old)=broken.at(old)*fault;
        map.setOptimizedPoses(broken,links);
        if(map.addLink(edge,true) || tip->hasLink(anchor) || tip->getWeight()!=weight ||
           map.getLocalOptimizedPoses()!=broken)
            throw std::runtime_error("an invalid historical graph was accepted or modified");
    }
    map.setOptimizedPoses(poses,links);
    if(!map.addLink(edge,true) || !tip->hasLink(anchor,Link::kGlobalClosure))
        throw std::runtime_error("verified leaf attachment failed");
    for(const auto & old : history)
        if(map.getLocalOptimizedPoses().at(old.first)!=old.second)
            throw std::runtime_error("attaching a single measured leaf changed the accepted graph");
    const auto final=map.getLocalOptimizedPoses();
    // Deliberately associate the same measured transform with a distant, wrong
    // anchor. This would create a cycle and must go through native optimization.
    const auto wrong=std::max_element(history.begin(),history.end(),[&](const auto & a,const auto & b) {
        return a.second.getDistanceSquared(history.at(anchor))<b.second.getDistanceSquared(history.at(anchor));
    })->first;
    if(map.addLink(Link(wrong,node,Link::kGlobalClosure,registration.transform(),registration.infMatrix()),true) ||
       tip->hasLink(wrong) || map.getLocalOptimizedPoses()!=final)
        throw std::runtime_error("contradictory second attachment passed graph validation");
    map.close();
    map.init(params,argv[1]);
    if(map.getLocalOptimizedPoses()!=final ||
       !map.getMemory()->getLinks(node,true).count(anchor))
        throw std::runtime_error("the measured edge and graph must survive reopening");
}

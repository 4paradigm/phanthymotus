// Run with a fresh database path. Hold a read transaction across addStatistics.
#include <rtabmap/core/DBDriverSqlite3.h>
#include <rtabmap/core/Statistics.h>
#include <sqlite3.h>
#include <chrono>
#include <thread>

int main(int argc,char **argv)
{
    if(argc!=2) return 2;
    rtabmap::DBDriverSqlite3 writer;
    if(!writer.openConnection(argv[1])) return 3;
    sqlite3 *reader=nullptr;
    if(sqlite3_open_v2(argv[1],&reader,SQLITE_OPEN_READONLY,nullptr)!=SQLITE_OK ||
       sqlite3_exec(reader,"BEGIN; SELECT * FROM Statistics;",nullptr,nullptr,nullptr)!=SQLITE_OK) return 4;
    std::thread release([reader](){
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        sqlite3_close(reader);
    });
    rtabmap::Statistics stats;
    stats.setRefImageId(1);
    stats.setStamp(1.);
    stats.addStatistic("Test/Lock/",1);
    try {writer.addStatistics(stats,false);}
    catch(...) {release.join(); return 5;}
    release.join();
    writer.closeConnection(false);
    if(!writer.openConnection(argv[1])) return 6;
    double stamp=0.;
    return writer.getStatistics(1,stamp).count("Test/Lock/")==1?0:7;
}

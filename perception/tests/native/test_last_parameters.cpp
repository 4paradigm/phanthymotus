#include <rtabmap/core/DBDriverSqlite3.h>

int main()
{
    rtabmap::DBDriverSqlite3 db;
    if (!db.openConnection(":memory:"))
        return 1;

    const rtabmap::ParametersMap first{{"Rtabmap/DetectionRate", "1"}};
    const rtabmap::ParametersMap last{{"Rtabmap/DetectionRate", "2"}};
    db.addInfoAfterRun(0, 0, 0, 0, 0, first);
    db.addInfoAfterRun(0, 0, 0, 0, 0, last);
    db.executeNoResult("UPDATE Info SET time_enter='2026-01-01 00:00:00'");
    return db.getLastParameters() == last ? 0 : 1;
}

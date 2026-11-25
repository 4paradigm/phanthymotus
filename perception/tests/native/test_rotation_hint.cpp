#include "RotationHintBuffer.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using motus::Quaternion;
using motus::RotationHintBuffer;

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string(__func__) + ": " + #condition); } while (false)

const std::int64_t BASE = 1800000000000000000LL;
const std::int64_t STEP = 100000000LL;
const char *FRAME = "camera_optical";

Quaternion yaw(double radians)
{
    return Quaternion{0.0, 0.0, std::sin(radians / 2.0), std::cos(radians / 2.0)};
}

void rotation(const Quaternion &q, double x, double y, double z, double w)
{
    const double sign = q.w * w < 0.0 ? -1.0 : 1.0;
    CHECK(std::abs(q.x * sign - x) < 1e-10);
    CHECK(std::abs(q.y * sign - y) < 1e-10);
    CHECK(std::abs(q.z * sign - z) < 1e-10);
    CHECK(std::abs(q.w * sign - w) < 1e-10);
}

void exact_stamp_and_identity()
{
    RotationHintBuffer buffer(FRAME);
    CHECK(buffer.add(BASE, FRAME, Quaternion{}, BASE));
    auto first = buffer.select(BASE, BASE);
    CHECK(first.has_orientation && !first.has_rotation);
    buffer.complete(first, true);
    // These captures collapse to the same double timestamp on Unix time.
    CHECK(double(BASE) == double(BASE + 1));
    CHECK(buffer.add(BASE + 1, FRAME, yaw(0.2), BASE + 1));
    CHECK(buffer.add(BASE + 2, FRAME, yaw(0.4), BASE + 2));
    auto second = buffer.select(BASE + 1, BASE + 2);
    CHECK(second.has_rotation);
    rotation(second.rotation, 0.0, 0.0, std::sin(0.1), std::cos(0.1));
    buffer.complete(second, true);
    auto third = buffer.select(BASE + 2, BASE + 2);
    CHECK(third.has_rotation);
    rotation(third.rotation, 0.0, 0.0, std::sin(0.1), std::cos(0.1));
}

void missing_and_late_never_match()
{
    RotationHintBuffer buffer(FRAME);
    CHECK(buffer.add(BASE + 1, FRAME, yaw(0.3), BASE));
    auto missing = buffer.select(BASE, BASE);
    CHECK(!missing.has_orientation && !missing.has_rotation);
    CHECK(!buffer.add(BASE, FRAME, Quaternion{}, BASE));
    buffer.complete(missing, true);
    auto future = buffer.select(BASE + 1, BASE + 1);
    CHECK(future.has_orientation && !future.has_rotation);
    CHECK(buffer.size() == 0);
    CHECK(buffer.stats().late == 1);
}

void dropped_and_failed_captures_keep_success_anchor()
{
    RotationHintBuffer buffer(FRAME);
    CHECK(buffer.add(BASE, FRAME, yaw(0.1), BASE));
    auto first = buffer.select(BASE, BASE);
    buffer.complete(first, true);
    // B arrives on the hint topic but its image is dropped by native ingress.
    CHECK(buffer.add(BASE + STEP, FRAME, yaw(0.3), BASE + STEP));
    CHECK(buffer.add(BASE + 2 * STEP, FRAME, yaw(0.6), BASE + 2 * STEP));
    auto failed = buffer.select(BASE + 2 * STEP, BASE + 2 * STEP);
    CHECK(failed.has_rotation);
    rotation(failed.rotation, 0.0, 0.0, std::sin(0.25), std::cos(0.25));
    buffer.complete(failed, false);
    CHECK(buffer.add(BASE + 3 * STEP, FRAME, yaw(0.8), BASE + 3 * STEP));
    auto recovered = buffer.select(BASE + 3 * STEP, BASE + 3 * STEP);
    CHECK(recovered.has_rotation);
    rotation(recovered.rotation, 0.0, 0.0, std::sin(0.35), std::cos(0.35));
    buffer.complete(recovered, true);
    CHECK(buffer.add(BASE + 4 * STEP, FRAME, yaw(0.9), BASE + 4 * STEP));
    auto next = buffer.select(BASE + 4 * STEP, BASE + 4 * STEP);
    rotation(next.rotation, 0.0, 0.0, std::sin(0.05), std::cos(0.05));
}

void successful_unhinted_capture_clears_anchor()
{
    RotationHintBuffer buffer(FRAME);
    CHECK(buffer.add(BASE, FRAME, yaw(0.1), BASE));
    buffer.complete(buffer.select(BASE, BASE), true);
    buffer.complete(buffer.select(BASE + STEP, BASE + STEP), true);
    CHECK(buffer.add(BASE + 2 * STEP, FRAME, yaw(0.6), BASE + 2 * STEP));
    auto next = buffer.select(BASE + 2 * STEP, BASE + 2 * STEP);
    CHECK(next.has_orientation && !next.has_rotation);
    buffer.complete(next, true);
    CHECK(buffer.add(BASE + 3 * STEP, FRAME, yaw(0.8), BASE + 3 * STEP));
    auto anchored = buffer.select(BASE + 3 * STEP, BASE + 3 * STEP);
    CHECK(anchored.has_rotation);
    rotation(anchored.rotation, 0.0, 0.0, std::sin(0.1), std::cos(0.1));
}

void bad_frame_quaternion_and_time_are_optional()
{
    RotationHintBuffer buffer(FRAME);
    CHECK(!buffer.add(BASE, "wrong", Quaternion{}, BASE));
    CHECK(!buffer.add(BASE, "", Quaternion{}, BASE));
    CHECK(!buffer.add(BASE, FRAME, Quaternion{0, 0, 0, 0}, BASE));
    CHECK(!buffer.add(BASE, FRAME, Quaternion{0, 0, 0, 2}, BASE));
    CHECK(!buffer.add(BASE, FRAME, Quaternion{0, 0, 0,
        std::numeric_limits<double>::infinity()}, BASE));
    CHECK(!buffer.add(BASE, FRAME, Quaternion{
        std::numeric_limits<double>::quiet_NaN(), 0, 0, 1}, BASE));
    CHECK(!buffer.add(0, FRAME, Quaternion{}, BASE));
    CHECK(!buffer.add(-1, FRAME, Quaternion{}, BASE));
    CHECK(!buffer.add(BASE, FRAME, Quaternion{}, 0));
    CHECK(!buffer.add(BASE, FRAME, Quaternion{}, BASE + 750000001LL));
    CHECK(!buffer.add(BASE + 50000001LL, FRAME, Quaternion{}, BASE));
    auto absent = buffer.select(BASE, BASE);
    CHECK(!absent.has_orientation && !absent.has_rotation);
    buffer.complete(absent, true);
    CHECK(buffer.add(BASE + STEP, FRAME, Quaternion{0, 0, 0, 1.0001}, BASE + STEP));
    auto valid = buffer.select(BASE + STEP, BASE + STEP);
    CHECK(valid.has_orientation);
    rotation(valid.orientation, 0, 0, 0, 1);
    CHECK(buffer.stats().rejected == 11);
}

void stale_selection_and_old_anchor_skip_rotation()
{
    RotationHintBuffer buffer(FRAME);
    CHECK(buffer.add(BASE, FRAME, Quaternion{}, BASE));
    auto stale = buffer.select(BASE, BASE + 750000001LL);
    CHECK(!stale.has_orientation && !stale.has_rotation);
    buffer.complete(stale, true);
    CHECK(buffer.add(BASE + 1000000000LL, FRAME, yaw(0.2), BASE + 1000000000LL));
    buffer.complete(buffer.select(BASE + 1000000000LL, BASE + 1000000000LL), true);
    CHECK(buffer.add(BASE + 1750000001LL, FRAME, yaw(0.8), BASE + 1750000001LL));
    auto long_gap = buffer.select(BASE + 1750000001LL, BASE + 1750000001LL);
    CHECK(long_gap.has_orientation && !long_gap.has_rotation);
}

void sign_and_noncommuting_rotation_direction()
{
    RotationHintBuffer buffer(FRAME);
    const double s = std::sqrt(0.5);
    // Reference is Rx(90 degrees); current is Ry(90 degrees).
    CHECK(buffer.add(BASE, FRAME, Quaternion{s, 0, 0, s}, BASE));
    buffer.complete(buffer.select(BASE, BASE), true);
    CHECK(buffer.add(BASE + STEP, FRAME, Quaternion{0, -s, 0, -s}, BASE + STEP));
    auto relative = buffer.select(BASE + STEP, BASE + STEP);
    CHECK(relative.has_rotation);
    // Rx(-90) * Ry(90), not the reversed multiplication order.
    rotation(relative.rotation, -0.5, 0.5, -0.5, 0.5);
    buffer.complete(relative, true);
    CHECK(buffer.add(BASE + 2 * STEP, FRAME, Quaternion{0, s, 0, s}, BASE + 2 * STEP));
    auto identity = buffer.select(BASE + 2 * STEP, BASE + 2 * STEP);
    CHECK(identity.has_rotation);
    rotation(identity.rotation, 0, 0, 0, 1);
}

void cache_is_bounded_and_duplicates_do_not_replace()
{
    RotationHintBuffer buffer(FRAME, 2);
    CHECK(buffer.add(BASE, FRAME, yaw(0.1), BASE));
    CHECK(!buffer.add(BASE, FRAME, yaw(0.9), BASE));
    CHECK(buffer.add(BASE + 1, FRAME, yaw(0.2), BASE));
    CHECK(buffer.add(BASE + 2, FRAME, yaw(0.3), BASE));
    CHECK(buffer.size() == 2);
    CHECK(!buffer.select(BASE, BASE).has_orientation);
    auto retained = buffer.select(BASE + 1, BASE + 1);
    rotation(retained.orientation, 0, 0, std::sin(0.1), std::cos(0.1));
    CHECK(buffer.stats().evicted == 1);
}

void reset_and_out_of_order_completion_cannot_restore_anchor()
{
    RotationHintBuffer buffer(FRAME);
    CHECK(buffer.add(BASE, FRAME, Quaternion{}, BASE));
    auto before_reset = buffer.select(BASE, BASE);
    buffer.reset();
    buffer.complete(before_reset, true);
    CHECK(buffer.size() == 0);
    CHECK(buffer.add(BASE - STEP, FRAME, yaw(0.3), BASE - STEP));
    auto first = buffer.select(BASE - STEP, BASE - STEP);
    CHECK(first.has_orientation && !first.has_rotation);
    buffer.complete(first, true);
    CHECK(buffer.add(BASE, FRAME, yaw(0.4), BASE));
    auto second = buffer.select(BASE, BASE);
    CHECK(second.has_rotation);
    buffer.complete(second, true);
    // A repeated/older decision cannot clear or rewind the accepted anchor.
    buffer.complete(buffer.select(BASE - STEP, BASE), true);
    buffer.complete(first, true);
    CHECK(buffer.add(BASE + STEP, FRAME, yaw(0.5), BASE + STEP));
    auto third = buffer.select(BASE + STEP, BASE + STEP);
    CHECK(third.has_rotation);
    rotation(third.rotation, 0, 0, std::sin(0.05), std::cos(0.05));
}

int main(int argc, char **argv)
{
    try {
        CHECK(argc == 2);
        const std::string name(argv[1]);
        if (name == "exact_stamp") exact_stamp_and_identity();
        else if (name == "missing_late") missing_and_late_never_match();
        else if (name == "dropped_failed") dropped_and_failed_captures_keep_success_anchor();
        else if (name == "clear_anchor") successful_unhinted_capture_clears_anchor();
        else if (name == "invalid") bad_frame_quaternion_and_time_are_optional();
        else if (name == "stale") stale_selection_and_old_anchor_skip_rotation();
        else if (name == "rotation") sign_and_noncommuting_rotation_direction();
        else if (name == "bounded") cache_is_bounded_and_duplicates_do_not_replace();
        else if (name == "reset") reset_and_out_of_order_completion_cannot_restore_anchor();
        else throw std::runtime_error("unknown test case");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}

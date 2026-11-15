#ifndef MOTUS_ROTATION_HINT_BUFFER_H
#define MOTUS_ROTATION_HINT_BUFFER_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

namespace motus {

struct Quaternion
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double w = 1.0;
};

// Optional registration hints, never pose observations. Caller serializes
// add/select/complete/reset; no lock is held while visual registration runs.
// A source/clock epoch change requires a new instance-private topic/session.
class RotationHintBuffer
{
public:
    struct Selection
    {
        std::int64_t stamp = 0;
        bool has_orientation = false;
        bool has_rotation = false;
        Quaternion orientation;
        Quaternion rotation;
        std::uint64_t generation = 0;
        std::uint64_t sequence = 0;
    };

    struct Stats
    {
        std::uint64_t received = 0;
        std::uint64_t rejected = 0;
        std::uint64_t late = 0;
        std::uint64_t evicted = 0;
        std::uint64_t frames = 0;
        std::uint64_t selected = 0;
        std::uint64_t supplied = 0;
    };

    explicit RotationHintBuffer(const std::string &frame, std::size_t capacity = 64) :
        frame_(frame), capacity_(std::max<std::size_t>(1, capacity))
    {}

    bool add(std::int64_t stamp, const std::string &frame,
             Quaternion orientation, std::int64_t now)
    {
        ++stats_.received;
        if (frame.empty() || frame != frame_ || !validTime(stamp, now) ||
            !normalize(orientation))
        {
            ++stats_.rejected;
            return false;
        }
        if (stamp <= watermark_)
        {
            ++stats_.late;
            return false;
        }
        if (!hints_.emplace(stamp, orientation).second)
        {
            ++stats_.rejected;
            return false;
        }
        if (hints_.size() > capacity_)
        {
            hints_.erase(hints_.begin());
            ++stats_.evicted;
        }
        return true;
    }

    Selection select(std::int64_t stamp, std::int64_t now)
    {
        Selection result;
        if (stamp <= 0 || stamp <= watermark_)
            return result;
        watermark_ = stamp;
        result.stamp = stamp;
        result.generation = generation_;
        result.sequence = ++sequence_;
        ++stats_.frames;
        const auto found = hints_.find(stamp);
        if (found != hints_.end() && validTime(stamp, now))
        {
            result.has_orientation = true;
            result.orientation = found->second;
            ++stats_.selected;
            if (anchor_stamp_ > 0 && stamp > anchor_stamp_ &&
                stamp - anchor_stamp_ <= MAX_AGE_NS)
            {
                result.rotation = relative(anchor_, result.orientation);
                result.has_rotation = true;
                ++stats_.supplied;
            }
        }
        // Never use a nearest timestamp or let a late callback repair a frame
        // whose visual update has already started. Retain future captures only.
        hints_.erase(hints_.begin(), hints_.upper_bound(stamp));
        return result;
    }

    void complete(const Selection &selection, bool success)
    {
        if (selection.stamp <= 0 || selection.generation != generation_ ||
            selection.sequence != sequence_ || selection.sequence <= completed_)
            return;
        completed_ = selection.sequence;
        if (!success)
            return; // Native pose still refers to the previous successful image.
        anchor_stamp_ = selection.has_orientation ? selection.stamp : 0;
        if (selection.has_orientation)
            anchor_ = selection.orientation;
    }

    void reset()
    {
        hints_.clear();
        watermark_ = anchor_stamp_ = 0;
        sequence_ = completed_ = 0;
        ++generation_; // An in-flight old completion cannot restore an anchor.
    }

    std::size_t size() const { return hints_.size(); }
    const Stats &stats() const { return stats_; }

private:
    // Same capture-age and future-time bounds as the calibrated IMU adapter.
    // Also bounds the relative rotation interval across failed registrations.
    static constexpr std::int64_t MAX_AGE_NS = 750000000LL;
    static constexpr std::int64_t MAX_FUTURE_NS = 50000000LL;

    static bool validTime(std::int64_t stamp, std::int64_t now)
    {
        if (stamp <= 0 || now <= 0)
            return false;
        return stamp <= now ? now - stamp <= MAX_AGE_NS : stamp - now <= MAX_FUTURE_NS;
    }

    static bool normalize(Quaternion &q)
    {
        if (!std::isfinite(q.x) || !std::isfinite(q.y) ||
            !std::isfinite(q.z) || !std::isfinite(q.w))
            return false;
        const double norm2 = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;
        if (!std::isfinite(norm2) || std::abs(norm2 - 1.0) > 0.002)
            return false;
        const double inverse = 1.0 / std::sqrt(norm2);
        q.x *= inverse;
        q.y *= inverse;
        q.z *= inverse;
        q.w *= inverse;
        return true;
    }

    static Quaternion relative(const Quaternion &from, const Quaternion &to)
    {
        // conjugate(from) * to: rotation of current camera into last camera.
        Quaternion q{
            from.w*to.x - from.x*to.w - from.y*to.z + from.z*to.y,
            from.w*to.y + from.x*to.z - from.y*to.w - from.z*to.x,
            from.w*to.z - from.x*to.y + from.y*to.x - from.z*to.w,
            from.w*to.w + from.x*to.x + from.y*to.y + from.z*to.z};
        normalize(q);
        return q;
    }

    const std::string frame_;
    const std::size_t capacity_;
    std::map<std::int64_t, Quaternion> hints_;
    std::int64_t watermark_ = 0;
    std::int64_t anchor_stamp_ = 0;
    Quaternion anchor_;
    std::uint64_t generation_ = 1;
    std::uint64_t sequence_ = 0;
    std::uint64_t completed_ = 0;
    Stats stats_;
};

} // namespace motus
#endif

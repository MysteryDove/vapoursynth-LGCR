#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sched.h>

namespace {

std::vector<int> parseCpuList(const std::string &text) {
    std::vector<int> result;
    size_t offset = 0;
    while (offset < text.size()) {
        const size_t comma = text.find(',', offset);
        const std::string item = text.substr(offset, comma - offset);
        const size_t dash = item.find('-');
        if (dash == std::string::npos) {
            result.push_back(std::stoi(item));
        } else {
            const int first = std::stoi(item.substr(0, dash));
            const int last = std::stoi(item.substr(dash + 1));
            if (last < first)
                throw std::runtime_error("descending CPU range");
            for (int cpu = first; cpu <= last; ++cpu)
                result.push_back(cpu);
        }
        if (comma == std::string::npos)
            break;
        offset = comma + 1;
    }
    if (result.empty())
        throw std::runtime_error("empty CPU list");
    return result;
}

void pinThread(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0)
        throw std::runtime_error("pthread_setaffinity_np failed for CPU " +
                                 std::to_string(cpu));
}

enum class Operation { Copy, Triad };

double measure(Operation operation, std::vector<double> &a,
               const std::vector<double> &b, const std::vector<double> &c,
               int threads, const std::vector<int> &cpus, double duration) {
    std::atomic<int> ready{0};
    std::atomic<bool> start{false}, stop{false};
    std::vector<uint64_t> iterations(threads, 0);
    std::vector<std::thread> workers;
    workers.reserve(threads);
    const size_t count = a.size();

    for (int worker = 0; worker < threads; ++worker) {
        workers.emplace_back([&, worker] {
            pinThread(cpus[worker]);
            const size_t begin = count * size_t(worker) / threads;
            const size_t end = count * size_t(worker + 1) / threads;
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire))
                std::this_thread::yield();
            uint64_t loops = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                if (operation == Operation::Copy) {
                    for (size_t i = begin; i < end; ++i)
                        a[i] = b[i];
                } else {
                    for (size_t i = begin; i < end; ++i)
                        a[i] = b[i] + 3.0 * c[i];
                }
                ++loops;
            }
            iterations[worker] = loops;
        });
    }
    while (ready.load(std::memory_order_acquire) != threads)
        std::this_thread::yield();
    const auto beginTime = std::chrono::steady_clock::now();
    start.store(true, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::duration<double>(duration));
    stop.store(true, std::memory_order_relaxed);
    for (auto &worker : workers)
        worker.join();
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - beginTime).count();

    long double bytes = 0.0;
    const int streams = operation == Operation::Copy ? 2 : 3;
    for (int worker = 0; worker < threads; ++worker) {
        const size_t begin = count * size_t(worker) / threads;
        const size_t end = count * size_t(worker + 1) / threads;
        bytes += static_cast<long double>(iterations[worker]) * (end - begin) *
                 sizeof(double) * streams;
    }
    return static_cast<double>(bytes / seconds / 1.0e9L);
}

} // namespace

int main(int argc, char **argv) {
    int threads = 1;
    size_t bytesPerArray = size_t{256} << 20;
    double duration = 2.0;
    std::string cpuText = "0";
    for (int i = 1; i < argc; ++i) {
        auto value = [&](const char *name) {
            if (++i == argc)
                throw std::runtime_error(std::string("missing value for ") + name);
            return std::string(argv[i]);
        };
        if (std::strcmp(argv[i], "--threads") == 0)
            threads = std::stoi(value("--threads"));
        else if (std::strcmp(argv[i], "--cpu-list") == 0)
            cpuText = value("--cpu-list");
        else if (std::strcmp(argv[i], "--bytes-mib") == 0)
            bytesPerArray = size_t(std::stoull(value("--bytes-mib"))) << 20;
        else if (std::strcmp(argv[i], "--duration") == 0)
            duration = std::stod(value("--duration"));
        else
            throw std::runtime_error(std::string("unknown option: ") + argv[i]);
    }
    const std::vector<int> cpus = parseCpuList(cpuText);
    if (threads < 1 || duration <= 0.0 || int(cpus.size()) < threads)
        throw std::runtime_error("invalid threads, duration, or CPU list");
    const size_t count = std::max<size_t>(1024, bytesPerArray / sizeof(double));
    std::vector<double> a(count, 0.0), b(count, 1.0), c(count, 2.0);
    const double copy = measure(Operation::Copy, a, b, c, threads, cpus, duration);
    const double triad = measure(Operation::Triad, a, b, c, threads, cpus, duration);
    std::cout << "{\"threads\":" << threads
              << ",\"cpu_list\":\"" << cpuText
              << "\",\"bytes_per_array\":" << count * sizeof(double)
              << ",\"copy_gbps\":" << copy
              << ",\"triad_gbps\":" << triad
              << ",\"sustainable_gbps\":" << std::max(copy, triad)
              << "}\n";
}

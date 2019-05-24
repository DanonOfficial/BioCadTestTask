#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <tuple>
#include <chrono>
#include <CL/cl.hpp>

constexpr double c = 1389.38757f;


int findDepth(const std::vector<uint32_t> &bonds, const std::vector<uint32_t> &index, size_t in, size_t key) {
    int depth = 4;

    for (uint32_t i = index[in]; i < (in == index.size() ? bonds.size() : index[in + 1]); i++) { // 2
        if (bonds[i] == key) {
            return 1;
        }
        for (uint32_t j = index[bonds[i]]; j < (in == index.size() ? bonds.size() : index[bonds[i] + 1]); j++) {
            if (bonds[j] == key) {
                return 2;
            }
            for (uint32_t k = index[bonds[j]]; k < (in == index.size() ? bonds.size() : index[bonds[j] + 1]); k++) {
                if (bonds[k] == key) {
                    depth = std::min(depth, 3);
                }
            }
        }
    }
    return depth;
}

double distance(double x1, double y1, double z1, double x2, double y2, double z2) {
    return sqrtf((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
}

std::string readKernel(const std::string &name) {
    std::string result;
    std::ifstream in(name);
    in.seekg(0, std::ios::end);
    result.reserve(in.tellg());
    in.seekg(0, std::ios::beg);

    result.assign((std::istreambuf_iterator<char>(in)),
                  std::istreambuf_iterator<char>());
    return result;
}

auto dataLoading() {
    std::ifstream in("atoms.txt");
    std::vector<float> atoms;
    float atom;
    while (in >> atom) {
        atoms.emplace_back(atom);
    }
    std::vector<float> charges;
    in = std::ifstream("charges.txt");
    float charge;
    while (in >> charge) {
        charges.emplace_back(charge);
    }
    std::vector<std::vector<size_t>> bonds(charges.size());
    std::vector<uint32_t> linedBonds;
    std::vector<uint32_t> indexingBonds(charges.size(), -1);

    in = std::ifstream("bonds.txt");
    uint32_t from, to;
    while (in >> from) {
        in >> to;
        bonds[from].emplace_back(to);
        bonds[to].emplace_back(from);
    }
    for (uint32_t i = 0; i < bonds.size(); i++) {
        indexingBonds[i] = linedBonds.size();
        linedBonds.insert(linedBonds.end(), bonds[i].cbegin(), bonds[i].cend());
    }
    return std::make_tuple(atoms, charges, linedBonds, indexingBonds, bonds);
}

void cpu() {
    using ms = std::chrono::milliseconds;
    std::vector<float> atoms, charges;
    std::vector<uint32_t> linedBonds, indexingBonds;
    std::vector<std::vector<size_t>> bonds;
    std::tie(atoms, charges, linedBonds, indexingBonds, bonds) = dataLoading();
    auto start = std::chrono::steady_clock::now();

    double E = 0.0f;
    std::vector<size_t> vertex;
    for (uint32_t i = 0; i < bonds.size(); i++) {
        for (uint32_t j = i + 1; j < bonds.size(); j++) {
            size_t depth = findDepth(linedBonds, indexingBonds, i, j);

            float f;
            if (depth < 3) {
                f = 0;
            } else {
                if (depth == 3) {
                    f = 0.5f;
                } else {
                    f = 1.f;
                }
            }
            float dist = distance(atoms[j * 3], atoms[j * 3 + 1], atoms[j * 3 + 2], atoms[i * 3], atoms[i * 3 + 1],
                                  atoms[i * 3 + 2]);
            size_t index = (2 * (bonds.size() - 1) - (i - 1)) * i / 2 + j - (i + 1);
            E += f * charges[i] * charges[j] / dist;
        }
    }
    E *= c;
    auto end = std::chrono::steady_clock::now();
    auto timeElapsed = end - start;
    std::cout << "Caclulation time: " << std::chrono::duration_cast<ms>(timeElapsed).count() << "ms" << std::endl;
    std::cout << "Energy: " << E << std::endl;
}

void gpu(size_t deviceIndex) {
    using ms = std::chrono::milliseconds;
    std::vector<float> atoms, charges;
    std::vector<uint32_t> linedBonds, indexingBonds;
    std::vector<std::vector<size_t>> bonds;
    std::tie(atoms, charges, linedBonds, indexingBonds, bonds) = dataLoading();
    std::vector<float> result(bonds.size() * (bonds.size() - 1) / 2);
    ///GPU
    std::vector<cl::Platform> allPlatforms;
    cl::Platform::get(&allPlatforms);
    if (allPlatforms.size() == 0) {
        std::cout << " No platforms found. Check OpenCL installation!" << std::endl;
        exit(1);
    }
    cl::Platform platform = allPlatforms[deviceIndex];
    std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    std::vector<cl::Device> allDevices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &allDevices);
    if (allDevices.size() == 0) {
        std::cout << " No devices found. Check OpenCL installation!" << std::endl;
        exit(1);
    }
    cl::Device device = allDevices[0];
    std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    cl::Context context({device});
    cl::Program::Sources sources;
    std::string kernelSrc = readKernel("kernel.cl");
    sources.push_back({kernelSrc.c_str(), kernelSrc.length()});

    cl::Program program(context, sources);
    if (program.build({device}) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
        exit(1);
    }
    cl::Buffer atomsBuf(context, CL_MEM_READ_ONLY, sizeof(float) * atoms.size());
    cl::Buffer chargesBuf(context, CL_MEM_READ_ONLY, sizeof(float) * charges.size());
    cl::Buffer bondsBuf(context, CL_MEM_READ_ONLY, sizeof(uint32_t) * linedBonds.size());
    cl::Buffer indexBuf(context, CL_MEM_READ_ONLY, sizeof(uint32_t) * indexingBonds.size());
    cl::Buffer resultBuf(context, CL_MEM_READ_WRITE, 4 * sizeof(float) * result.size());

    cl::CommandQueue queue(context, device);

    queue.enqueueWriteBuffer(atomsBuf, CL_TRUE, 0, sizeof(float) * atoms.size(), atoms.data());
    queue.enqueueWriteBuffer(chargesBuf, CL_TRUE, 0, sizeof(float) * charges.size(), charges.data());
    queue.enqueueWriteBuffer(bondsBuf, CL_TRUE, 0, sizeof(uint32_t) * linedBonds.size(), linedBonds.data());
    queue.enqueueWriteBuffer(indexBuf, CL_TRUE, 0, sizeof(uint32_t) * indexingBonds.size(), indexingBonds.data());


    cl::Kernel kernel = cl::Kernel(program, "solve");
    kernel.setArg(0, chargesBuf);
    kernel.setArg(1, bondsBuf);
    kernel.setArg(2, indexBuf);
    kernel.setArg(3, atomsBuf);
    kernel.setArg(4, resultBuf);
    kernel.setArg(5, (uint32_t) charges.size());
    kernel.setArg(6, (uint32_t) bonds.size());
    kernel.setArg(7, (uint32_t) (charges.size() * charges.size()));
    unsigned int workGroupSize = 128;
    unsigned int globalWorkSize = (charges.size() * charges.size());
    queue.enqueueNDRangeKernel(kernel, 0, cl::NDRange(globalWorkSize), cl::NDRange(workGroupSize));

    auto start = std::chrono::steady_clock::now();

    queue.finish();

    queue.enqueueReadBuffer(resultBuf, CL_TRUE, 0, sizeof(float) * result.size(), result.data());
    double E = 0.0f;

    for (auto &i: result) {
        E += i;
    }
    E *= c;
    auto end = std::chrono::steady_clock::now();
    auto timeElapsed = end - start;
    std::cout << "Caclulation time: " << std::chrono::duration_cast<ms>(timeElapsed).count() << "ms" << std::endl;
    std::cout << "Energy: " << E << std::endl;

}

int main(int argc, char **argv) {
    std::cout << "Using CPU" << std::endl;
    cpu();
    size_t deviceIndex = std::stoi(argv[1]);
    std::cout << "Using GPU" << std::endl;
    gpu(deviceIndex);
    return 0;
}
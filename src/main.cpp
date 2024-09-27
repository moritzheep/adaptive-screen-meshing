#include <iostream>
#include <pmp/io/io.h>
#include "smp/algorithms/Integration.h"
#include "utility/ArgumentParser.h"
#include "PhotometricRemeshing.h"

void print_help_message()
{
    std::cout << "Options\n";
    std::cout << "  -n <path-to-normal-map>      = Path to the normal maps (as .exr file)\n";
    std::cout << "  -e <approximation-error>     = Desired approximation error (orthographic: pixels, perspective: mm)\n";
    std::cout << "  -t <path-to-triangle-mesh>   = Path to write the output mesh to (we recommend using .obj files)\n"; 
    std::cout << "\n";
    std::cout << "  -m <path-to-foreground-mask> = Path to foreground mask (as b/w .png file, optional) \n";
    std::cout << "  -l <lower-limit>             = Minimal allowed edge length (optional, default: 1px)\n";
    std::cout << "  -h <higher-limit>            = Maximal allowed edge length (optional, default: 100px)\n";

    std::cout << "\nYou can switch from orthographic to perspective projection by supplying intrinsics\n";
    std::cout << "  | -x  0 -u |\n";
    std::cout << "  |  0 -y -v |\n";
    std::cout << "  |  0  0  1 |\n";
    std::cout << "\nPlease Note: In orthographic mode, all lengths (approximation error, minimal and maximal lengths) are in pixels.\nIn perspective mode, we use millimeters instead.\n";
}


int main(int argc, char *argv[])
{
    ArgumentParser parser(argc, argv);

    if (!parser.has_arguments())
    {
        print_help_message();
        return 0;
    }

    if (!parser.has_argument('n'))
    {
        std::cout << "Error: No Normal Map provided!\n\n";
        print_help_message();
        return 1;
    }

    if (!parser.has_argument('t'))
    {
        std::cout << "Warning: No output Mesh provided!\n\n";
    }

    cv::Mat normals = cv::imread(parser.get_argument('n'), cv::IMREAD_UNCHANGED);

    // Discard alpha if neccessary
    if (normals.channels() == 4)
    {
        cv::cvtColor(normals, normals, cv::COLOR_BGRA2BGR);
    }
    else if (normals.channels() != 3)
    {
        std::cout << "Error: Normal Map must have three/four channels.\n";
        return 1;
    }

    int height = normals.rows;
    int width = normals.cols;

    // Load Mask
    cv::Mat mask;

    if (parser.has_argument('m'))
    {
        mask = cv::imread(parser.get_argument('m'), cv::IMREAD_GRAYSCALE);
    }
    else
    {
        mask = 255 * cv::Mat::ones(height, width, CV_8UC1);

        for (int v = 0; v != height; ++v)
        {
            mask.at<uchar>(v, 0) = 0;
            mask.at<uchar>(v, width - 1) = 0;
        }
        for (int u = 0; u != width; ++u)
        {
            mask.at<uchar>(0, u) = 0;
            mask.at<uchar>(height - 1, u) = 0;
        }
    }

    if (mask.empty())
    {
        std::cout << "Error: Could not read provided foreground mask!\n";
        std::cout << "The foreground mask must be a black-and-white .png image.\n";
        return 1;
    }

    // Remap normals
    for (int v = 0; v != height; ++v)
    {
        for (int u = 0; u != width; ++u)
        {
            if (mask.at<uchar>(v, u) > 127)
            {
                normals.at<cv::Vec3f>(v, u) = cv::Vec3f(1. - 2. * normals.at<cv::Vec3f>(v, u)[0], 1. - 2. * normals.at<cv::Vec3f>(v, u)[1], 1. - 2. * normals.at<cv::Vec3f>(v, u)[2]);
            }
        }
    }

    // Perspective Mode
    if (parser.has_argument('x'))
    {
        float ax = std::stod(parser.get_argument('x'));
        float ay = parser.has_argument('y') ? std::stod(parser.get_argument('y')) : ax;

        float u0 = parser.has_argument('u') ? std::stod(parser.get_argument('u')) : static_cast<float>(width) / 2.;
        float v0 = parser.has_argument('v') ? std::stod(parser.get_argument('v')) : static_cast<float>(height) / 2.;

        float scale = 1. / std::sqrt(std::abs(ax * ay)); // Approximately the size of one pixel

        // Load remeshing values
        float l_min = parser.has_argument('l') ? std::stof(parser.get_argument('l')) : scale;
        float l_max = parser.has_argument('h') ? std::stof(parser.get_argument('h')) : 100. * scale;
        float approx_error = parser.has_argument('e') ? (std::stof(parser.get_argument('e')) / 1000.) : 1. * scale;

        // Run remeshing
        PhotometricRemeshing<pmp::Perspective> remesher(normals.clone(), mask, pmp::Perspective(ax, -ay, u0, v0));
        remesher.run(l_min, l_max, approx_error);

        // Perform integration
        pmp::Integration<double, pmp::Perspective> integrator(remesher.mesh(), normals.clone(), mask, pmp::Perspective(ax, -ay, u0, v0));
        integrator.run();

        // Transform mesh to world-coordinates
        for (auto v : remesher.mesh().vertices())
        {
            auto pos = remesher.mesh().position(v);

            pos[0] -= u0;
            pos[0] /= ax;

            pos[1] -= v0;
            pos[1] /= ay;

            remesher.mesh().position(v) = pos;
        }

        // Remove slivers
        int slivers = integrator.remove_slivers();

        if (slivers > 0)
        {
            std::cout << "Removed " << slivers << " slivers\n";
        }

        if (parser.has_argument('t'))
        {
            pmp::write(remesher.mesh(), parser.get_argument('t'));
        }
    }
    // Orthographic Mode
    else
    {
        // Load remeshing values
        float l_min = parser.has_argument('l') ? std::stof(parser.get_argument('l')) : 1;
        float l_max = parser.has_argument('h') ? std::stof(parser.get_argument('h')) : 100.;
        float approx_error = parser.has_argument('e') ? std::stof(parser.get_argument('e')) : 1.;

        PhotometricRemeshing<pmp::Orthographic> remesher(normals.clone(), mask);
        remesher.run(l_min, l_max, approx_error);

        // Perform integration
        pmp::Integration<double, pmp::Orthographic> integrator(remesher.mesh(), normals.clone(), mask);
        integrator.run();

        // Remove slivers
        int slivers = integrator.remove_slivers();

        if (slivers > 0)
        {
            std::cout << "Removed " << slivers << " slivers\n";
        }

        if (parser.has_argument('t'))
        {
            pmp::write(remesher.mesh(), parser.get_argument('t'));
        }
    }
}

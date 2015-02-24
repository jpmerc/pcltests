#include <pcl/io/pcd_io.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/shot.h>
#include <pcl/features/fpfh.h>
#include <iostream>
#include <string>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/time.h>

int
main(int argc, char** argv)
{

    typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ColorHandlerT;

    // Object for storing the SHOT descriptors for the scene.
    pcl::PointCloud<pcl::PointXYZ>::Ptr sceneKeypoints(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr sceneDescriptors(new pcl::PointCloud<pcl::FPFHSignature33>());
    // Object for storing the SHOT descriptors for the model.
    pcl::PointCloud<pcl::PointXYZ>::Ptr modelKeypoints(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr modelDescriptors(new pcl::PointCloud<pcl::FPFHSignature33>());

    pcl::PointCloud<pcl::PointXYZ>::Ptr model_aligned(new pcl::PointCloud<pcl::PointXYZ>());

    // Read the already computed descriptors from disk.
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *sceneKeypoints) != 0)
    {
        return -1;
    }
    if (pcl::io::loadPCDFile<pcl::FPFHSignature33>(argv[2], *sceneDescriptors) != 0)
    {
        return -1;
    }
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[3], *modelKeypoints) != 0)
    {
        return -1;
    }
    if (pcl::io::loadPCDFile<pcl::FPFHSignature33>(argv[4], *modelDescriptors) != 0)
    {
        return -1;
    }


    pcl::SampleConsensusPrerejective<pcl::PointXYZ,pcl::PointXYZ,pcl::FPFHSignature33> align;
    align.setInputSource (modelKeypoints);
    align.setSourceFeatures (modelDescriptors);
    align.setInputTarget (sceneKeypoints);
    align.setTargetFeatures (sceneDescriptors);
    align.setMaximumIterations (10000); // Number of RANSAC iterations
    align.setNumberOfSamples (8); // Number of points to sample for generating/prerejecting a pose
    align.setCorrespondenceRandomness (4); // Number of nearest features to use
    align.setSimilarityThreshold (0.9f); // Polygonal edge length similarity threshold
    const float leaf = 0.05f;
    align.setMaxCorrespondenceDistance (1.5f * leaf); // Inlier threshold
    align.setInlierFraction (0.25f); // Required inlier fraction for accepting a pose hypothesis

    align.align (*model_aligned);


    Eigen::Matrix4f transformation = align.getFinalTransformation ();
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("Inliers: %i/%i\n", align.getInliers ().size (), modelKeypoints->size ());

    // Show alignment
    pcl::visualization::PCLVisualizer visu("Alignment");
    visu.addPointCloud (sceneKeypoints, ColorHandlerT (sceneKeypoints, 0.0, 255.0, 0.0), "scene");
    visu.addPointCloud (model_aligned, ColorHandlerT (model_aligned, 255.0, 0.0, 0.0), "object_aligned");
    visu.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "object_aligned");
    visu.spin ();

}

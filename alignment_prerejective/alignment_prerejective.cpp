#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>

// Types
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;

double computeCloudResolution(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
{
    double resolution = 0.0;
    int numberOfPoints = 0;
    int nres;
    std::vector<int> indices(2);
    std::vector<float> squaredDistances(2);
    pcl::search::KdTree<pcl::PointXYZ> tree;
    tree.setInputCloud(cloud);

    for (size_t i = 0; i < cloud->size(); ++i)
    {
        if (! pcl_isfinite((*cloud)[i].x))
            continue;

        // Considering the second neighbor since the first is the point itself.
        nres = tree.nearestKSearch(i, 2, indices, squaredDistances);
        if (nres == 2)
        {
            resolution += sqrt(squaredDistances[1]);
            ++numberOfPoints;
        }
    }
    if (numberOfPoints != 0)
        resolution /= numberOfPoints;

    return resolution;
}

pcl::PointCloud<pcl::PointNormal>::Ptr computeNormalsFromPC(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

    // Estimate the normals.
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);

    normalEstimation.setInputCloud(cloud);
    normalEstimation.setKSearch(10);
    normalEstimation.setSearchMethod(kdtree);
    normalEstimation.compute(*normals);

    pcl::PointCloud<pcl::PointNormal>::Ptr merged_cloud (new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*cloud, *normals, *merged_cloud);

    return merged_cloud;

}


// Align a rigid object to a scene with clutter and occlusions
int
main (int argc, char **argv)
{
    // Point clouds
    PointCloudT::Ptr object (new PointCloudT);
    PointCloudT::Ptr object_aligned (new PointCloudT);
    PointCloudT::Ptr scene (new PointCloudT);
    FeatureCloudT::Ptr object_features (new FeatureCloudT);
    FeatureCloudT::Ptr scene_features (new FeatureCloudT);

    // Get input object and scene
    if (argc != 3)
    {
        pcl::console::print_error ("Syntax is: %s object.pcd scene.pcd\n", argv[0]);
        return (1);
    }

    // Load object and scene
    pcl::console::print_highlight ("Loading point clouds...\n");
    pcl::PointCloud<pcl::PointXYZ>::Ptr model_xyz (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_xyz (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile(argv[1], *model_xyz);
    pcl::io::loadPCDFile(argv[2], *scene_xyz);
    object = computeNormalsFromPC(model_xyz);
    scene = computeNormalsFromPC(scene_xyz);

    double model_cloud_resolution = computeCloudResolution(model_xyz);
    double scene_cloud_resolution = computeCloudResolution(scene_xyz);
    std::cout << "Model Cloud Resolution = " << model_cloud_resolution << std::endl;
    std::cout << "Scene Cloud Resolution = " << scene_cloud_resolution << std::endl;


    // Downsample
    pcl::console::print_highlight ("Downsampling...\n");
    pcl::VoxelGrid<PointNT> grid;
    const float leaf = 0.01f;
    grid.setLeafSize (leaf, leaf, leaf);
    grid.setInputCloud (object);
    grid.filter (*object);
    grid.setInputCloud (scene);
    grid.filter (*scene);

    // Estimate normals for scene
//    pcl::console::print_highlight ("Estimating scene normals...\n");
//    pcl::NormalEstimationOMP<PointNT,PointNT> nest;
//    nest.setRadiusSearch (0.01);
//    nest.setInputCloud (scene);
//    nest.compute (*scene);

    // Estimate features
    pcl::console::print_highlight ("Estimating features...\n");
    FeatureEstimationT fest;
    fest.setRadiusSearch (0.025);
    fest.setInputCloud (object);
    fest.setInputNormals (object);
    fest.compute (*object_features);
    fest.setInputCloud (scene);
    fest.setInputNormals (scene);
    fest.compute (*scene_features);

    // Perform alignment
    pcl::console::print_highlight ("Starting alignment...\n");
    pcl::SampleConsensusPrerejective<PointNT,PointNT,FeatureT> align;
    align.setInputSource (object);
    align.setSourceFeatures (object_features);
    align.setInputTarget (scene);
    align.setTargetFeatures (scene_features);
    align.setMaximumIterations (50000); // Number of RANSAC iterations
    align.setNumberOfSamples (3); // Number of points to sample for generating/prerejecting a pose
    align.setCorrespondenceRandomness (10); // Number of nearest features to use
    align.setSimilarityThreshold (0.75f); // Polygonal edge length similarity threshold
    align.setMaxCorrespondenceDistance (4.0f * leaf); // Inlier threshold
    align.setInlierFraction (0.95); // Required inlier fraction for accepting a pose hypothesis
    {
        pcl::ScopeTime t("Alignment");
        align.align (*object_aligned);
    }




    // Print results
    printf ("\n");
    Eigen::Matrix4f transformation = align.getFinalTransformation ();
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("Inliers: %i/%i\n", align.getInliers ().size (), object->size ());

    if (align.hasConverged ())
    {
        // Show alignment
        pcl::visualization::PCLVisualizer visu("Alignment");
        visu.addPointCloud (scene, ColorHandlerT (scene, 0.0, 255.0, 0.0), "scene");
        visu.addPointCloud (object_aligned, ColorHandlerT (object_aligned, 255.0, 0.0, 0.0), "object_aligned");
        visu.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "object_aligned");
        visu.spin ();
    }
    else
    {
        pcl::console::print_error ("Alignment failed!\n");
        return (1);
    }

    return (0);
}

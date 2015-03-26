#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/bilateral.h>
#include <pcl/filters/fast_bilateral_omp.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/surface/bilateral_upsampling.h>


typedef pcl::PointXYZRGB PointT;
typedef pcl::visualization::PointCloudColorHandlerRGBField<PointT> ColorHandlerRGB;

pcl::PointCloud<PointT>::Ptr filter(pcl::PointCloud<PointT>::Ptr cloud){

    pcl::ScopeTime t("Bilateral Filter");

    pcl::PointCloud<PointT>::Ptr cloud_out(new pcl::PointCloud<PointT>);

//    pcl::FastBilateralFilterOMP<PointT> bilateral_filter;
//    bilateral_filter.setInputCloud (cloud);
//    bilateral_filter.setSigmaS (5);
//    bilateral_filter.setSigmaR (0.005f);
//    bilateral_filter.filter (*cloud_out);

    pcl::BilateralUpsampling<PointT, PointT> bilateral_upsampling;
    bilateral_upsampling.setInputCloud (cloud);
    bilateral_upsampling.setWindowSize(3);
    bilateral_upsampling.setSigmaColor(2.5f);
    bilateral_upsampling.setSigmaDepth(.5f);
    bilateral_upsampling.setProjectionMatrix(bilateral_upsampling.KinectVGAProjectionMatrix);
    bilateral_upsampling.process(*cloud_out);




    return cloud_out;

}

int main (int argc, char** argv)
{
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>),
            cloud_filtered (new pcl::PointCloud<PointT>),
            cloud_filtered_translated (new pcl::PointCloud<PointT>);


    pcl::io::loadPCDFile("../cloud.pcd", *cloud);


    cloud_filtered = filter(cloud);


    // Transform PC to visualize in same Viewer
    Eigen::Matrix4f cloud_translation;
    cloud_translation.setIdentity();
    cloud_translation(0,3) = 2; //x translation to compare with other mls
    pcl::transformPointCloud(*cloud_filtered, *cloud_filtered_translated, cloud_translation);




    pcl::visualization::PCLVisualizer visu("Bilateral Filter");

    visu.addPointCloud (cloud, ColorHandlerRGB(cloud), "cloud");
    visu.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

    visu.addPointCloud (cloud_filtered_translated, ColorHandlerRGB(cloud_filtered_translated), "cloud_filtered");
    visu.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_filtered");

    visu.spin ();

    // pcl::PCDWriter writer;
    // writer.write ("table_scene_mug_stereo_textured_hull.pcd", *cloud_hull, false);

    return (0);
}

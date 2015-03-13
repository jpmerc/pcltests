// PCL specific includes
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <vtkRenderWindow.h>
#include <pcl/common/transforms.h>
#include <stdlib.h>
#include <sstream>
#include <stdio.h>
#include <time.h>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/time.h>
#include <pcl/filters/fast_bilateral_omp.h>
#include <pcl/surface/bilateral_upsampling.h>
#include <pcl/filters/median_filter.h>

typedef pcl::PointXYZRGB PointT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointT> ColorHandlerT;
using namespace std;

boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));

void initPCLViewer(){
    //PCL Viewer
    //pclViewer->registerKeyboardCallback (keyboardEventOccurred, (void*)&pclViewer);
    pclViewer->setBackgroundColor (0, 0, 0);
    pclViewer->initCameraParameters ();
    pclViewer->setCameraPosition(0,0,0,0,0,1,0,-1,0);
    vtkSmartPointer<vtkRenderWindow> renderWindow = pclViewer->getRenderWindow();
    renderWindow->SetSize(800,450);
    renderWindow->Render();

}


pcl::PointCloud<PointT>::Ptr downsample(pcl::PointCloud<PointT>::Ptr cloud, double voxel_size){
    pcl::ScopeTime t("Voxel_grid Filtering");
    pcl::PointCloud<PointT>::Ptr returnCloud(new pcl::PointCloud<PointT>);
    pcl::VoxelGrid<PointT> grid;
    const float leaf = voxel_size;
    grid.setLeafSize (leaf, leaf, leaf);
    grid.setInputCloud (cloud);
    grid.filter (*returnCloud);

    return returnCloud;
}

pcl::PointCloud<PointT>::Ptr computeSurfaceNormals(pcl::PointCloud<PointT>::Ptr cloud)
{
    pcl::ScopeTime t("Normals");

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

    // Estimate the normals.
    pcl::NormalEstimationOMP<PointT, pcl::Normal> normalEstimation;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);

    normalEstimation.setInputCloud(cloud);
    //normalEstimation.setKSearch(10);
    normalEstimation.setRadiusSearch(0.025);
    normalEstimation.setSearchMethod(kdtree);
    normalEstimation.compute(*normals);

    pcl::PointCloud<PointT>::Ptr merged_cloud (new pcl::PointCloud<PointT>);
    pcl::concatenateFields(*cloud, *normals, *merged_cloud);


    return merged_cloud;

}





Eigen::Matrix4f align_icp(pcl::PointCloud<PointT>::Ptr src_cloud, pcl::PointCloud<PointT>::Ptr target_cloud, Eigen::Matrix4f initial_transform, double maxCorrespondanceDistance){

    pcl::ScopeTime t("ICP");

    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setInputSource(src_cloud);
    icp.setInputTarget(target_cloud);
    icp.setMaxCorrespondenceDistance(maxCorrespondanceDistance);
    icp.setMaximumIterations(40);
    //icp.setRANSACOutlierRejectionThreshold(0.02);
    //icp.setUseReciprocalCorrespondences(true);

    pcl::PointCloud<PointT>::Ptr Final(new pcl::PointCloud<PointT>());
    icp.align(*Final,initial_transform);
    Eigen::Matrix4f icp_transform = icp.getFinalTransformation();
    double fitness_score = icp.getFitnessScore();
    cout << "ICP Transformation Score = " << fitness_score << endl;

    return icp_transform;
}





pcl::PointCloud<PointT>::Ptr downsample(pcl::PointCloud<PointT>::Ptr cloud, double voxel_size){
    pcl::ScopeTime t("Voxel_grid Filtering");
    pcl::PointCloud<PointT>::Ptr returnCloud(new pcl::PointCloud<PointT>);
    pcl::VoxelGrid<PointT> grid;
    const float leaf = voxel_size;
    grid.setLeafSize (leaf, leaf, leaf);
    grid.setInputCloud (cloud);
    grid.filter (*returnCloud);

    return returnCloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr fillColors(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud){

    for(int i = 0; i < cloud->points.size(); i++){
        pcl::PointXYZRGB pt = cloud->at(i);
        pt.r = 0;
        pt.g = 255;
        pt.b = 0;
        cloud->at(i) = pt;
    }

    return cloud;

}


pcl::PointCloud<pcl::PointXYZRGB>::Ptr filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud){

    pcl::ScopeTime t("Bilateral Filter");

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::FastBilateralFilterOMP<pcl::PointXYZRGB> bilateral_filter;
    bilateral_filter.setInputCloud (cloud);
    bilateral_filter.setSigmaS (5);
    bilateral_filter.setSigmaR (0.005f);
    bilateral_filter.filter (*cloud_out);

//    pcl::MedianFilter<pcl::PointXYZRGB> median_filter;
//    median_filter.setInputCloud (cloud);
//    median_filter.setWindowSize (7);
//    median_filter.setMaxAllowedMovement (0.02);
//    median_filter.filter (*cloud_out);

    return cloud_out;

}


int main (int argc, char** argv){


    pcl::PointCloud<PointT>::Ptr in1(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr in2(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr container_model(new pcl::PointCloud<PointT>);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr in1_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr in2_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr container_model_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>);


    pcl::io::loadPCDFile("../device1_1.pcd", *in1_xyzrgb);
    pcl::io::loadPCDFile("../device2_1.pcd", *in2_xyzrgb);
    pcl::io::loadPCDFile("../container_model.pcd", *container_model_xyzrgb);

   // container_model_xyzrgb = fillColors(container_model_xyzrgb);

    initPCLViewer();


    // Sampling
    
    
    // Features
    
    
    // Correspondences Calculation (brute force)
    
    
    // Correspondence Rejection (RANSAC) 
    // Select 3 feature pairs randomly
    // Find Transform
    // Calculate number of (feature-feature distance <= threshold)
    // After n iterations, keep the best transform
    // Reject the feature pairs for which the point to point distance is bigger than threshold in the aligned clouds
    
    
    // Singular Value Decomposition (SVD) with Least Squares Error (fine alignment with only good features)
    
    
    




    pclViewer->addPointCloud (in1_xyzrgb, ColorHandlerT(in1_xyzrgb, 0.0, 255.0, 0.0), "2");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "2");










    while (!pclViewer->wasStopped()) {
        pclViewer->spinOnce (100);
    }
    return 0;
}

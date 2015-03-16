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
#include <angles/angles.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/time.h>
#include <pcl/filters/fast_bilateral_omp.h>
#include <pcl/surface/bilateral_upsampling.h>
#include <pcl/filters/median_filter.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_rejection_median_distance.h>
#include <pcl/registration/correspondence_rejection_surface_normal.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/correspondence_rejection_features.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/harris_3d.h>


/*
  This file aligns point clouds with each other by rejecting correspondences with RANSAC
  and then fine tuning the alignement with SVD. The correspondences are calculated
  by brute force.
*/


typedef pcl::PointXYZRGBNormal PointT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointT> ColorHandlerT;
using namespace std;

boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer2 (new pcl::visualization::PCLVisualizer ("3D Viewer"));


struct CorrespondanceResults {
  Eigen::Matrix4f transformation;
  pcl::Correspondences correspondences;
  pcl::Correspondences initial_correspondences;
} ;


void initPCLViewer(){
    //PCL Viewer
    //pclViewer->registerKeyboardCallback (keyboardEventOccurred, (void*)&pclViewer);
    pclViewer->setBackgroundColor (0, 0, 0);
    pclViewer->initCameraParameters ();
    pclViewer->setCameraPosition(0,0,0,0,0,1,0,-1,0);
    vtkSmartPointer<vtkRenderWindow> renderWindow = pclViewer->getRenderWindow();
    renderWindow->SetSize(800,450);
    renderWindow->Render();

    pclViewer2->setBackgroundColor (0, 0, 0);
    pclViewer2->initCameraParameters ();
    pclViewer2->setCameraPosition(0,0,0,0,0,1,0,-1,0);
    vtkSmartPointer<vtkRenderWindow> renderWindow2 = pclViewer2->getRenderWindow();
    renderWindow2->SetSize(800,450);
    renderWindow2->Render();

}


pcl::PointCloud<PointT>::Ptr computeSurfaceNormals(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    pcl::ScopeTime t("Normals");

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

    // Estimate the normals.
    pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> normalEstimation;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);

    normalEstimation.setInputCloud(cloud);
    //normalEstimation.setKSearch(10);
    normalEstimation.setRadiusSearch(0.025);
    normalEstimation.setSearchMethod(kdtree);
    normalEstimation.compute(*normals);

    pcl::PointCloud<PointT>::Ptr merged_cloud (new pcl::PointCloud<PointT>);
    pcl::concatenateFields(*cloud, *normals, *merged_cloud);


    return merged_cloud;

}

pcl::PointCloud<PointT>::Ptr computeSurfaceNormals_withKeypoints(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<PointT>::Ptr searchSurface)
{
    pcl::ScopeTime t("Normals");

    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

    // Estimate the normals.
    pcl::NormalEstimationOMP<PointT, pcl::Normal> normalEstimation;
    pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);

    normalEstimation.setInputCloud(cloud);
    //normalEstimation.setKSearch(10);
    normalEstimation.setRadiusSearch(0.025);
    normalEstimation.setSearchMethod(kdtree);
    normalEstimation.setSearchSurface(searchSurface);
    normalEstimation.compute(*normal_cloud);

    pcl::PointCloud<PointT>::Ptr merged_cloud (new pcl::PointCloud<PointT>);
    pcl::concatenateFields(*cloud, *normal_cloud, *merged_cloud);

    return merged_cloud;

}


pcl::PointCloud<pcl::FPFHSignature33>::Ptr computeFPFH (pcl::PointCloud<PointT>::Ptr cloud){

    pcl::ScopeTime t("FPFH");

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr features(new pcl::PointCloud<pcl::FPFHSignature33>);
    pcl::search::KdTree<PointT>::Ptr tree;

    pcl::FPFHEstimationOMP<PointT, PointT, pcl::FPFHSignature33>::Ptr fpfh_est(new pcl::FPFHEstimationOMP<PointT, PointT, pcl::FPFHSignature33>);
    fpfh_est->setInputCloud (cloud);
    fpfh_est->setInputNormals (cloud);
    fpfh_est->setSearchMethod (tree);
    //fpfh_est->setKSearch(10);
    fpfh_est->setRadiusSearch (0.05);
    //fpfh_est->setIndices();

    fpfh_est->compute (*features);

    return features;
}

pcl::PointCloud<pcl::FPFHSignature33>::Ptr computeFPFH_withKeypoints (pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<PointT>::Ptr searchSurface){

    pcl::ScopeTime t("FPFH");

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr features(new pcl::PointCloud<pcl::FPFHSignature33>);
    pcl::search::KdTree<PointT>::Ptr tree;

    pcl::FPFHEstimationOMP<PointT, PointT, pcl::FPFHSignature33>::Ptr fpfh_est(new pcl::FPFHEstimationOMP<PointT, PointT, pcl::FPFHSignature33>);
    fpfh_est->setInputCloud (cloud);
    fpfh_est->setInputNormals (searchSurface);
    fpfh_est->setSearchMethod (tree);
    //fpfh_est->setKSearch(10);
    fpfh_est->setRadiusSearch (0.05);
    //fpfh_est->setIndices();
    fpfh_est->setSearchSurface(searchSurface);

    fpfh_est->compute (*features);

    return features;
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

pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsample(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, double voxel_size){
    pcl::ScopeTime t("Voxel_grid Filtering");
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr returnCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::VoxelGrid<pcl::PointXYZRGB> grid;
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
    bilateral_filter.setSigmaR (0.01f);
    bilateral_filter.filter (*cloud_out);

//    pcl::MedianFilter<pcl::PointXYZRGB> median_filter;
//    median_filter.setInputCloud (cloud);
//    median_filter.setWindowSize (7);
//    median_filter.setMaxAllowedMovement (0.02);
//    median_filter.filter (*cloud_out);

    return cloud_out;

}


CorrespondanceResults coarse_alignment(pcl::PointCloud<PointT>::ConstPtr src_cloud, pcl::PointCloud<pcl::FPFHSignature33>::Ptr src_features, pcl::PointCloud<PointT>::Ptr target_cloud, pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_features){

    pcl::ScopeTime t("Ransac coarse alignment");

    CorrespondanceResults results;

    // NOTE
    // Direct correspondence estimation (default) searches for correspondences in cloud B for every point in cloud A .
    // “Reciprocal” correspondence estimation searches for correspondences from cloud A to cloud B, and from B to A and only use the intersection.

    // Correspondences Calculation (brute force)
    pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> corr_estimator;
    pcl::Correspondences correspondences;
    corr_estimator.setInputSource(src_features);
    corr_estimator.setInputTarget(target_features);
    corr_estimator.determineCorrespondences(correspondences);
    results.initial_correspondences = correspondences;


    pcl::registration::CorrespondenceRejectorMedianDistance rej_median;
    pcl::Correspondences correspondences_median;
    rej_median.setInputSource<PointT>(src_cloud);
    rej_median.setInputTarget<PointT>(target_cloud);
    rej_median.setMedianFactor(2);
    rej_median.getRemainingCorrespondences(correspondences, correspondences_median);


//    pcl::registration::CorrespondenceRejectorSurfaceNormal rej_normals;
//    pcl::Correspondences correspondences_normals;
//    rej_normals.initializeDataContainer <PointT, PointT> ();
//    rej_normals.setInputSource<PointT>(src_cloud);
//    rej_normals.setInputTarget<PointT>(target_cloud);
//    rej_normals.setInputNormals<PointT,PointT>(src_cloud);
//    rej_normals.setTargetNormals<PointT,PointT>(target_cloud);
//    rej_normals.setThreshold(angles::from_degrees(30));

//    rej_median.getRemainingCorrespondences(correspondences_median, correspondences_normals);


    // Correspondence Rejection (RANSAC)
    // Select 3 feature pairs randomly
    // Find Transform
    // Calculate number of (feature-feature distance <= threshold)
    // After n iterations, keep the best transform
    // Reject the feature pairs for which the point to point distance is bigger than threshold in the aligned clouds
    pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> corr_rejector;
    corr_rejector.setInputSource(src_cloud);
    //corr_rejector.setInputCorrespondences(correspondences);
    corr_rejector.setInputTarget(target_cloud);
    corr_rejector.setMaximumIterations(150);
    corr_rejector.setInlierThreshold(0.05);
    corr_rejector.getRemainingCorrespondences(correspondences_median, results.correspondences);

    results.transformation  = corr_rejector.getBestTransformation();

    return results;

}


pcl::PointCloud<PointT>::Ptr computeISSKeypoints(pcl::PointCloud<PointT>::Ptr cloud)
{
    pcl::ScopeTime t("ISS Keypoints");

    pcl::PointCloud<PointT>::Ptr keypoints(new pcl::PointCloud<PointT>);

    pcl::ISSKeypoint3D<PointT, PointT> detector;
    detector.setInputCloud(cloud);
    pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
    detector.setSearchMethod(kdtree);
    double resolution = 0.01;

    // Set the radius of the spherical neighborhood used to compute the scatter matrix.
    detector.setSalientRadius(6 * resolution);
    // Set the radius for the application of the non maxima supression algorithm.
    detector.setNonMaxRadius(4 * resolution);
    // Set the minimum number of neighbors that has to be found while applying the non maxima suppression algorithm.
    detector.setMinNeighbors(5);
    // Set the upper bound on the ratio between the second and the first eigenvalue.
    detector.setThreshold21(0.975);
    // Set the upper bound on the ratio between the third and the second eigenvalue.
    detector.setThreshold32(0.975);
    // Set the number of prpcessing threads to use. 0 sets it to automatic.
    detector.setNumberOfThreads(4);

    detector.setNormalRadius (4 * resolution);
    detector.setBorderRadius (1 * resolution);

    detector.compute(*keypoints);


    return keypoints;

}


pcl::PointCloud<PointT>::Ptr computeHarris3DKeypoints(pcl::PointCloud<PointT>::Ptr cloud)
{
    pcl::ScopeTime t("Harris 3D Keypoints");
    pcl::PointCloud<PointT>::Ptr keypoints(new pcl::PointCloud<PointT>);


    pcl::HarrisKeypoint3D<PointT,pcl::PointXYZI> harris3D;
       // harris3D->setNonMaxSupression (true);
    //    harris3D->setInputCloud();


    //pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI, PointT> detector;
//    detector.setInputCloud(cloud);
//    detector.setNormals(cloud);
//    pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
//    detector.setSearchMethod(kdtree);
//    detector.compute(*keypoints);

    return keypoints;

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

   // in1_xyzrgb = filter(in1_xyzrgb);
  //  in2_xyzrgb = filter(in2_xyzrgb);
   // container_model_xyzrgb = filter(container_model_xyzrgb);


   // container_model_xyzrgb = fillColors(container_model_xyzrgb);

    initPCLViewer();


    // Sampling
    in1_xyzrgb = downsample(in1_xyzrgb, 0.01);
    in2_xyzrgb = downsample(in2_xyzrgb, 0.01);
    container_model_xyzrgb = downsample(container_model_xyzrgb, 0.005);


    // Compute normals
    in1 = computeSurfaceNormals(in1_xyzrgb);
    in2 = computeSurfaceNormals(in2_xyzrgb);
    container_model = computeSurfaceNormals(container_model_xyzrgb);

    // Keypoints
    pcl::PointCloud<PointT>::Ptr in1_keypoints = computeHarris3DKeypoints(in1);
    pcl::PointCloud<PointT>::Ptr in2__keypoints = computeHarris3DKeypoints(in2);
    pcl::PointCloud<PointT>::Ptr container_model_keypoints = computeHarris3DKeypoints(container_model);

    // Keypoints Normals
    in1_keypoints = computeSurfaceNormals_withKeypoints(in1_keypoints, in1);
    in2__keypoints = computeSurfaceNormals_withKeypoints(in2__keypoints, in2);
    container_model_keypoints = computeSurfaceNormals_withKeypoints(container_model_keypoints, container_model);

    // Features
//    pcl::PointCloud<pcl::FPFHSignature33>::Ptr in1_fpfh = computeFPFH(in1);
//    pcl::PointCloud<pcl::FPFHSignature33>::Ptr in2_fpfh = computeFPFH(in2);
//    pcl::PointCloud<pcl::FPFHSignature33>::Ptr container_model_fpfh = computeFPFH(container_model);

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr in1_fpfh = computeFPFH_withKeypoints(in1_keypoints, in1);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr in2_fpfh = computeFPFH_withKeypoints(in2__keypoints, in2);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr container_model_fpfh = computeFPFH_withKeypoints(container_model_keypoints, container_model);

    
    // Coarse Alignment with RANSAC
//    CorrespondanceResults coarse_results = coarse_alignment(container_model, container_model_fpfh, in1, in1_fpfh);
     CorrespondanceResults coarse_results = coarse_alignment(container_model_keypoints, container_model_fpfh, in1_keypoints, in1_fpfh);

    
    // Singular Value Decomposition (SVD) with Least Squares Error (fine alignment with only good features)
    pcl::registration::TransformationEstimationSVD<PointT, PointT> svd;
    Eigen::Matrix4f svd_transformation;
//    svd.estimateRigidTransformation(*container_model, *in1, coarse_results.correspondences, svd_transformation);
    svd.estimateRigidTransformation(*container_model_keypoints, *in1_keypoints, coarse_results.correspondences, svd_transformation);


    // Transform point clouds and show them
    pcl::PointCloud<PointT>::Ptr container_model_ransac(new pcl::PointCloud<PointT>);
    pcl::transformPointCloud(*container_model, *container_model_ransac, coarse_results.transformation);

    pcl::PointCloud<PointT>::Ptr container_model_svd(new pcl::PointCloud<PointT>);
    pcl::transformPointCloud(*container_model, *container_model_svd, svd_transformation);

    
    pclViewer->addPointCloud (in1, ColorHandlerT(in1, 255.0, 0.0, 0.0), "2");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "2");

    pclViewer->addPointCloud (container_model, ColorHandlerT(container_model, 0.0, 255.0, 0.0), "model");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "model");

    pclViewer->addPointCloud (container_model_ransac, ColorHandlerT(container_model_ransac, 0.0, 255.0, 0.0), "ransac");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "ransac");


//    pclViewer->addPointCloud (container_model_svd, ColorHandlerT(container_model_svd, 0.0, 0.0, 255.0), "svd");
//    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "svd");

    pclViewer->addPointCloud (container_model_keypoints, ColorHandlerT(container_model_keypoints, 0.0, 0.0, 255.0), "key");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "key");

    pclViewer->addCorrespondences<PointT>(container_model_keypoints, in1_keypoints, coarse_results.initial_correspondences, "corr1");



    //Keypoints Visualization
    pclViewer2->addPointCloud (container_model, ColorHandlerT(container_model, 255.0, 0.0, 0.0), "1");
    pclViewer2->addPointCloud (container_model_keypoints, ColorHandlerT(container_model_keypoints, 0.0, 0.0, 255.0), "2");
    pclViewer2->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "2");





    while (!pclViewer->wasStopped()) {
        pclViewer->spinOnce (100);
    }
    return 0;
}

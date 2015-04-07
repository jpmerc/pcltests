#include <stdlib.h>
#include <sstream>
#include <stdio.h>

// PCL specific includes
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <vtkRenderWindow.h>

#include <Eigen/Eigen>
#include <Eigen/Geometry>
//#include <angles/angles.h>

#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include <pcl/common/angles.h>

#include <pcl/keypoints/iss_3d.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>

#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_rejection_median_distance.h>
#include <pcl/registration/correspondence_rejection_surface_normal.h>
#include <pcl/registration/icp.h>

#include <pcl/filters/sampling_surface_normal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/filters/project_inliers.h>

#include <pcl/surface/convex_hull.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/mls.h>

#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/crf_normal_segmentation.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <boost/thread.hpp>
#include <vtkPolyLine.h>



typedef pcl::PointXYZRGBNormal PointT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointT> ColorHandlerT;
typedef pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> ColorHandlerRGB;
using namespace std;

boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
//boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer2 (new pcl::visualization::PCLVisualizer ("3D Viewer 2"));
boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer3 (new pcl::visualization::PCLVisualizer ("3D Viewer 3"));
boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer4 (new pcl::visualization::PCLVisualizer ("3D Viewer 4"));
boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer5 (new pcl::visualization::PCLVisualizer ("3D Viewer 5"));

double _distanceThreshold = 3.0;
double _pointColorThreshold = 3.0;
double _regionColorThreshold = 3.0;
double _minClusterSize = 20;


struct CorrespondenceResults {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix4f coarse_transformation;
    Eigen::Matrix4f fine_transformation;
    double fine_transformation_score;
    pcl::Correspondences correspondences;
    pcl::Correspondences initial_correspondences;
    pcl::PointCloud<PointT>::Ptr scene_cloud;
    pcl::PointCloud<PointT>::Ptr model_cloud;
    pcl::PointCloud<PointT>::Ptr scene_keypoints;
    pcl::PointCloud<PointT>::Ptr model_keypoints;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr scene_features;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr model_features;
    pcl::PointCloud<PointT>::Ptr model_coarse_aligned;
    pcl::PointCloud<PointT>::Ptr model_fine_aligned;
} ;

std::vector<CorrespondenceResults*> _AlignmentResults;
std::vector<pcl::PointCloud<PointT>::Ptr > segmented_clouds;

int l_count = 0;
int cloud_index = 0;
void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void);

void initPCLViewer(){
    //PCL Viewer
    //pclViewer->registerKeyboardCallback (keyboardEventOccurred, (void*)&pclViewer);
    pclViewer->setBackgroundColor (0, 0, 0);
    pclViewer->initCameraParameters ();
    pclViewer->setCameraPosition(0,0,0,0,0,1,0,-1,0);
    vtkSmartPointer<vtkRenderWindow> renderWindow = pclViewer->getRenderWindow();
    renderWindow->SetSize(800,450);
    renderWindow->Render();

    //    pclViewer2->setBackgroundColor (0, 0, 0);
    //    pclViewer2->initCameraParameters ();
    //    pclViewer2->setCameraPosition(0,0,0,0,0,1,0,-1,0);
    //    vtkSmartPointer<vtkRenderWindow> renderWindow2 = pclViewer2->getRenderWindow();
    //    renderWindow2->SetSize(800,450);
    //    renderWindow2->Render();

    pclViewer3->setBackgroundColor (0, 0, 0);
    pclViewer3->initCameraParameters ();
    pclViewer3->setCameraPosition(0,0,0,0,0,1,0,-1,0);
    vtkSmartPointer<vtkRenderWindow> renderWindow3 = pclViewer3->getRenderWindow();
    renderWindow3->SetSize(800,450);
    renderWindow3->Render();

    pclViewer4->setBackgroundColor (0, 0, 0);
    pclViewer4->initCameraParameters ();
    pclViewer4->setCameraPosition(0,0,0,0,0,1,0,-1,0);
    vtkSmartPointer<vtkRenderWindow> renderWindow4 = pclViewer4->getRenderWindow();
    renderWindow4->SetSize(800,450);
    renderWindow4->Render();

    pclViewer5->setBackgroundColor (0, 0, 0);
    pclViewer5->initCameraParameters ();
    pclViewer5->setCameraPosition(0,0,0,0,0,1,0,-1,0);
    pclViewer5->registerKeyboardCallback (keyboardEventOccurred, (void*)&pclViewer5);
    vtkSmartPointer<vtkRenderWindow> renderWindow5 = pclViewer5->getRenderWindow();
    renderWindow5->SetSize(800,450);
    renderWindow5->Render();

}

void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void)
{
    l_count = l_count + 1;
    if(l_count < 2){
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
        if (event.getKeySym () == "s"){

            viewer->removePointCloud("seg");
            viewer->addPointCloud (segmented_clouds.at(cloud_index), ColorHandlerT(segmented_clouds.at(cloud_index), 255.0, 255.0, 0.0), "seg");
            viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "seg");

            cloud_index++;
            if(cloud_index >= segmented_clouds.size()) cloud_index = 0;

        }
    }
    else{
        l_count = 0;
    }

}

void showTransform(std::vector<CorrespondenceResults*> *results, int index){

    CorrespondenceResults *corr = results->at(index);

    pclViewer->addPointCloud (corr->scene_cloud, ColorHandlerT(corr->scene_cloud, 255.0, 0.0, 0.0), "scene");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "scene");

    pclViewer->addPointCloud (corr->model_cloud, ColorHandlerT(corr->model_cloud, 0.0, 255.0, 0.0), "model");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "model");

    pclViewer->addPointCloud (corr->model_coarse_aligned, ColorHandlerT(corr->model_coarse_aligned, 0.0, 255.0, 0.0), "ransac");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "ransac");

    pclViewer->addPointCloud (corr->model_fine_aligned, ColorHandlerT(corr->model_fine_aligned, 0.0, 0.0, 255.0), "icp");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "icp");

    //    pclViewer->addPointCloud (corr->scene_keypoints, ColorHandlerT(corr->scene_keypoints, 255.0, 255.0, 0.0), "scene_key");
    //    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_key");

    pclViewer->addPointCloud (corr->model_keypoints, ColorHandlerT(corr->model_keypoints, 0.0, 0.0, 255.0), "model_key");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "model_key");

    pclViewer->addCorrespondences<PointT>(corr->model_keypoints, corr->scene_keypoints, corr->correspondences, "corr1");

}


void saveClouds(std::vector<CorrespondenceResults*> *results, int index){

    CorrespondenceResults *corr = results->at(index);

    pcl::io::savePCDFileASCII<PointT>(std::string("../customAlignment_scene.pcd"),  *corr->scene_cloud);
    pcl::io::savePCDFileASCII<PointT>(std::string("../customAlignment_coarse.pcd"), *corr->model_coarse_aligned);
    pcl::io::savePCDFileASCII<PointT>(std::string("../customAlignment_fine.pcd"),   *corr->model_fine_aligned);

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
    fpfh_est->setNumberOfThreads(2);

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

pcl::PointCloud<PointT>::Ptr normalSampling(pcl::PointCloud<PointT>::Ptr cloud){
    pcl::ScopeTime t("Normal Sampling");
    pcl::PointCloud<PointT>::Ptr returnCloud(new pcl::PointCloud<PointT>);

    pcl::SamplingSurfaceNormal<PointT> sampler;
    sampler.setRatio(0.1);
    //sampler.setSample(15);
    sampler.setInputCloud(cloud);
    sampler.filter (*returnCloud);

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





void coarse_alignment(CorrespondenceResults *data){

    pcl::ScopeTime t("Ransac coarse alignment");

    // NOTE
    // Direct correspondence estimation (default) searches for correspondences in cloud B for every point in cloud A .
    // “Reciprocal” correspondence estimation searches for correspondences from cloud A to cloud B, and from B to A and only use the intersection.

    // Correspondences Calculation (brute force)
    pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> corr_estimator;
    pcl::Correspondences correspondences;
    corr_estimator.setInputSource(data->model_features);
    corr_estimator.setInputTarget(data->scene_features);
    corr_estimator.determineCorrespondences(correspondences);
    data->initial_correspondences = correspondences;
    std::cout << "Initial Correspondences = " << data->initial_correspondences.size()  << std::endl;


    pcl::registration::CorrespondenceRejectorMedianDistance rej_median;
    pcl::Correspondences correspondences_median;
    rej_median.setInputSource<PointT>(data->model_keypoints);
    rej_median.setInputTarget<PointT>(data->scene_keypoints);
    rej_median.setMedianFactor(2);
    rej_median.getRemainingCorrespondences(correspondences, correspondences_median);
    std::cout << "Median Correspondences = " << correspondences_median.size()  << std::endl;

    //    pcl::registration::CorrespondenceRejectorSurfaceNormal rej_normals;
    //    pcl::Correspondences correspondences_normals;
    //    rej_normals.initializeDataContainer <PointT, PointT> ();
    //    rej_normals.setInputSource<PointT>(src_cloud);
    //    rej_normals.setInputTarget<PointT>(target_cloud);
    //    rej_normals.setInputNormals<PointT,PointT>(src_cloud);
    //    rej_normals.setTargetNormals<PointT,PointT>(target_cloud);
    //    rej_normals.setThreshold(pcl::deg2rad(45.));
    //    rej_median.getRemainingCorrespondences(correspondences_median, correspondences_normals);


    // Correspondence Rejection (RANSAC)
    // Select 3 feature pairs randomly
    // Find Transform
    // Calculate number of (feature-feature distance <= threshold)
    // After n iterations, keep the best transform
    // Reject the feature pairs for which the point to point distance is bigger than threshold in the aligned clouds
    pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> corr_rejector;
    corr_rejector.setInputSource(data->model_keypoints);
    //corr_rejector.setInputCorrespondences(correspondences);
    corr_rejector.setInputTarget(data->scene_keypoints);
    corr_rejector.setMaximumIterations(150);
    corr_rejector.setInlierThreshold(0.03);
    corr_rejector.getRemainingCorrespondences(correspondences_median, data->correspondences);

    data->coarse_transformation  = corr_rejector.getBestTransformation();

    // Transform point cloud
    pcl::PointCloud<PointT>::Ptr container_model_ransac(new pcl::PointCloud<PointT>);
    pcl::transformPointCloud(*data->model_cloud, *container_model_ransac, data->coarse_transformation);
    data->model_coarse_aligned = container_model_ransac;
}


pcl::PointCloud<PointT>::Ptr computeISSKeypoints(pcl::PointCloud<PointT>::Ptr cloud)
{
    pcl::ScopeTime t("ISS Keypoints");

    pcl::PointCloud<PointT>::Ptr cloud_downsampled = downsample(cloud, 0.015);


    pcl::PointCloud<PointT>::Ptr keypoints(new pcl::PointCloud<PointT>);

    pcl::ISSKeypoint3D<PointT, PointT, PointT> detector;
    detector.setInputCloud(cloud_downsampled);
    pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
    detector.setSearchMethod(kdtree);
    double resolution = 0.01;

    detector.setNormals(cloud_downsampled);

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
    detector.setNumberOfThreads(2);

    // detector.setNormalRadius (4 * resolution);
    //detector.setBorderRadius (4 * resolution);

    detector.compute(*keypoints);


    return keypoints;

}



void align_icp(CorrespondenceResults *data){

    pcl::ScopeTime t("ICP");

    //    typedef pcl::registration::TransformationEstimationPointToPlane<PointT, PointT> PointToPlane;
    //    boost::shared_ptr<PointToPlane> point_to_plane(new PointToPlane);

    // downsample the input pointclouds
    pcl::PointCloud<PointT>::Ptr src_downsampled = downsample(data->model_cloud, 0.02);
    pcl::PointCloud<PointT>::Ptr target_downsampled = downsample(data->scene_cloud, 0.02);

    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setInputSource(src_downsampled);
    icp.setInputTarget(target_downsampled);
    icp.setMaxCorrespondenceDistance(0.05);
    icp.setMaximumIterations(40);
    //icp.setTransformationEstimation(point_to_plane);

    pcl::PointCloud<PointT>::Ptr Final(new pcl::PointCloud<PointT>());
    icp.align(*Final,data->coarse_transformation);
    data->fine_transformation = icp.getFinalTransformation();
    data->fine_transformation_score = icp.getFitnessScore();
    cout << "ICP Transformation Score = " << data->fine_transformation_score << endl;

    // Transform
    pcl::PointCloud<PointT>::Ptr container_model_icp(new pcl::PointCloud<PointT>);
    pcl::transformPointCloud(*data->model_cloud, *container_model_icp, data->fine_transformation);
    data->model_fine_aligned = container_model_icp;

}

void ISS_thread(pcl::PointCloud<PointT>::Ptr in1, pcl::PointCloud<PointT>::Ptr container_model){

    pcl::ScopeTime t("Transformation with ISS Keypoints");

    CorrespondenceResults* iss(new CorrespondenceResults);
    iss->scene_cloud = in1;
    iss->model_cloud = container_model;

    iss->scene_keypoints = computeISSKeypoints(iss->scene_cloud);
    iss->model_keypoints = computeISSKeypoints(iss->model_cloud);

    iss->scene_keypoints = computeSurfaceNormals_withKeypoints(iss->scene_keypoints, iss->scene_cloud);
    iss->model_keypoints = computeSurfaceNormals_withKeypoints(iss->model_keypoints, iss->model_cloud);

    iss->scene_features = computeFPFH_withKeypoints(iss->scene_keypoints, iss->scene_cloud);
    iss->model_features = computeFPFH_withKeypoints(iss->model_keypoints, iss->model_cloud);

    coarse_alignment(iss);
    align_icp(iss);

    _AlignmentResults.push_back(iss);
}

void Sampling_thread(pcl::PointCloud<PointT>::Ptr scene_cloud, pcl::PointCloud<PointT>::Ptr model_cloud){

    pcl::ScopeTime t("Transformation with ISS Keypoints");

    CorrespondenceResults* sampling(new CorrespondenceResults);
    sampling->scene_cloud = scene_cloud;
    sampling->model_cloud = model_cloud;

    sampling->scene_keypoints = downsample(sampling->scene_cloud, 0.02);
    sampling->model_keypoints = downsample(sampling->model_cloud, 0.02);

    sampling->scene_keypoints = computeSurfaceNormals_withKeypoints(sampling->scene_keypoints, sampling->scene_cloud);
    sampling->model_keypoints = computeSurfaceNormals_withKeypoints(sampling->model_keypoints, sampling->model_cloud);

    sampling->scene_features = computeFPFH_withKeypoints(sampling->scene_keypoints, sampling->scene_cloud);
    sampling->model_features = computeFPFH_withKeypoints(sampling->model_keypoints, sampling->model_cloud);

    coarse_alignment(sampling);
    align_icp(sampling);

    _AlignmentResults.push_back(sampling);
}

int getBestTransformIndex(std::vector<CorrespondenceResults*> *transforms){

    double smallestScore = 999999;
    int smallestScore_index = -1;
    for(int i=0; i < transforms->size(); i++){
        double icp_score = transforms->at(i)->fine_transformation_score;
        if(icp_score < smallestScore){
            smallestScore = icp_score;
            smallestScore_index = i;
        }
    }

    return smallestScore_index;
}

pcl::PointCloud<PointT>::Ptr extractPlane(pcl::PointCloud<PointT>::Ptr cloud){

    pcl::PointCloud<PointT>::Ptr returned_cloud(new pcl::PointCloud<PointT>);

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inlierIndices(new pcl::PointIndices);

    // Find the plane coefficients from the model
    pcl::SACSegmentation<PointT> segmentation;
    segmentation.setInputCloud(cloud);
    segmentation.setModelType(pcl::SACMODEL_PLANE);
    segmentation.setMethodType(pcl::SAC_RANSAC);
    segmentation.setDistanceThreshold(0.02);
    segmentation.setOptimizeCoefficients(true);
    segmentation.segment(*inlierIndices, *coefficients);

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inlierIndices);
    extract.filter (*returned_cloud);

    return returned_cloud;

}

pcl::PointCloud<PointT>::Ptr getCentroid(pcl::PointCloud<PointT>::Ptr cloud){
    pcl::PointCloud<PointT>::Ptr centroid_cloud (new pcl::PointCloud<PointT>);
    Eigen::Vector4f c;
    pcl::compute3DCentroid<PointT>(*cloud,c);
    PointT pt;
    pt.x = c[0];
    pt.y = c[1];
    pt.z = c[2];
    centroid_cloud->push_back(pt);
    return centroid_cloud;
}

pcl::PointCloud<PointT>::Ptr getCentroid2D(pcl::PointCloud<PointT>::Ptr cloud){

    pcl::PointCloud<PointT>::Ptr bottom_plane = extractPlane(cloud);

    pcl::PointCloud<PointT>::Ptr centroid_cloud (new pcl::PointCloud<PointT>);
    Eigen::Vector4f c;
    pcl::compute3DCentroid<PointT>(*bottom_plane,c);
    PointT pt;
    pt.x = c[0];
    pt.y = c[1];
    pt.z = c[2];
    centroid_cloud->push_back(pt);
    return centroid_cloud;
}

pcl::PointCloud<PointT>::Ptr shrinkCloud(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<PointT>::Ptr centroid, double shrink_distance){

    // Theory
    // The equation of a 3d line is [xt;yt;zt] = [x1;y1;z1] + t [x2-x1;y2-y1;z2-z1]
    // In this case, 't' is a ratio between the distance to shrink and the distance between P2 (centroid) and P1 (pointcloud points)

    pcl::PointCloud<PointT>::Ptr shrinked_cloud (new pcl::PointCloud<PointT>);

    PointT centroid_pt = centroid->at(0);
    Eigen::Vector3f centroid_vec(centroid_pt.x, centroid_pt.y, centroid_pt.z);

    for(int i=0; i < cloud->size(); i++){

        PointT pointcloud_pt = cloud->at(i);
        Eigen::Vector3f pointcloud_vec(pointcloud_pt.x, pointcloud_pt.y, pointcloud_pt.z);

        // Calculate the difference vector and the distance between the points
        Eigen::Vector3f diff = centroid_vec - pointcloud_vec;
        double distance_between_points = diff.squaredNorm();

        // Calculate the distance ratio (shrink_distance over the distance between the points)
        double ratio = shrink_distance / distance_between_points;

        // Calculate the new coordinates of the point
        Eigen::Vector3f new_coords = pointcloud_vec + ratio * diff;

        // Assign the new coordinates to the original point
        pointcloud_pt.x = new_coords[0];
        pointcloud_pt.y = new_coords[1];
        pointcloud_pt.z = new_coords[2];
        shrinked_cloud->push_back(pointcloud_pt);

    }

    return shrinked_cloud;
}

pcl::PointCloud<PointT>::Ptr cropAndSegmentScene(pcl::PointCloud<PointT>::Ptr scene_cloud, pcl::PointCloud<PointT>::Ptr model_cloud, bool shrink_hull, bool set3D){

    pcl::ScopeTime t("Hull");

    pcl::ConvexHull<PointT> hull;
    pcl::PointCloud<PointT>::Ptr surface_hull (new pcl::PointCloud<PointT>);
    hull.setInputCloud(model_cloud);
    if(set3D) hull.setDimension(3);
    else hull.setDimension(2);
    // hull.setAlpha(3);
    std::vector<pcl::Vertices> polygons;
    hull.reconstruct(*surface_hull, polygons);

    std::cout << "Vertices : " << polygons.size() <<  std::endl;
    std::cout << "Hull Cloud : " << surface_hull->size() <<  std::endl;

    pclViewer3->addPointCloud (surface_hull, ColorHandlerT(surface_hull, 255.0, 255.0, 0.0), "hull");
    pclViewer3->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "hull");

    //    pclViewer3->addPointCloud (scene_cloud, ColorHandlerT(scene_cloud, 255.0, 0.0, 0.0), "scene");
    //    pclViewer3->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "scene");

    pcl::PointCloud<PointT>::Ptr projected_points(new pcl::PointCloud<PointT>);
    if(!set3D){
        // Find the coefficients of the hull plane and project points of the scene on it
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inlierIndices(new pcl::PointIndices);

        // Find the plane coefficients from the model
        pcl::SACSegmentation<PointT> segmentation;
        segmentation.setInputCloud(surface_hull);
        segmentation.setModelType(pcl::SACMODEL_PLANE);
        segmentation.setMethodType(pcl::SAC_RANSAC);
        segmentation.setDistanceThreshold(0.02);
        segmentation.setOptimizeCoefficients(true);
        segmentation.segment(*inlierIndices, *coefficients);

        // Project Scene Cloud in 2D plane of the Hull
        pcl::ProjectInliers<PointT> proj;
        proj.setModelType (pcl::SACMODEL_PLANE);
        proj.setInputCloud (scene_cloud);
        proj.setModelCoefficients (coefficients);
        proj.filter (*projected_points);

        pclViewer3->addPointCloud (projected_points, ColorHandlerT(projected_points, 0.0, 255.0, 255.0), "projection");
        pclViewer3->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "projection");
    }

    if(shrink_hull){
        pcl::PointCloud<PointT>::Ptr centroid_cloud = getCentroid(model_cloud);
        pcl::PointCloud<PointT>::Ptr shrinked_hull = shrinkCloud(surface_hull, centroid_cloud, 0.015);
        surface_hull = shrinked_hull;

        pclViewer3->addPointCloud (shrinked_hull, ColorHandlerT(shrinked_hull, 0.0, 255.0, 0.0), "shrinked_hull");
        pclViewer3->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "shrinked_hull");
    }

    pcl::PointCloud<PointT>::Ptr objects (new pcl::PointCloud<PointT>);
    pcl::CropHull<PointT> bb_filter2;

    if(set3D){
        bb_filter2.setDim(3);
        bb_filter2.setInputCloud(scene_cloud);
        bb_filter2.setHullIndices(polygons);
        bb_filter2.setHullCloud(surface_hull);
        bb_filter2.filter(*objects);
    }
    else{
        std::vector<int> indices;
        bb_filter2.setDim(2);
        bb_filter2.setInputCloud(projected_points);
        bb_filter2.setHullIndices(polygons);
        bb_filter2.setHullCloud(surface_hull);
        bb_filter2.filter(indices);

        for(int i=0; i < indices.size(); i++){
            objects->push_back( scene_cloud->at( indices.at(i)) );
        }
    }

    pclViewer3->addPointCloud (objects, ColorHandlerT(objects, 255.0, 0.0, 255.0), "scene_cropped");
    pclViewer3->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "scene_cropped");

    return objects;
}

pcl::PointCloud<PointT>::Ptr smoothPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud){

    pcl::ScopeTime t("Moving Least Squares");

    pcl::PointCloud<PointT>::Ptr cloud_smoothed2(new pcl::PointCloud<PointT>);
    pcl::MovingLeastSquares<pcl::PointXYZRGB, PointT> mls2;
    mls2.setInputCloud (cloud);
    mls2.setSearchRadius (0.04);
    mls2.setPolynomialFit (true);
    mls2.setPolynomialOrder (2);
    //    mls2.setUpsamplingMethod (pcl::MovingLeastSquares<pcl::PointXYZRGB, PointT>::SAMPLE_LOCAL_PLANE);
    //    mls2.setUpsamplingRadius (0.01);
    //    mls2.setUpsamplingStepSize (0.005);
    mls2.process (*cloud_smoothed2);

    cloud_smoothed2 = downsample(cloud_smoothed2, 0.01);

    pcl::PointCloud<PointT>::Ptr cloud_smoothed2_translated(new pcl::PointCloud<PointT>);
    Eigen::Matrix4f cloud_translation;
    cloud_translation.setIdentity();
    cloud_translation(0,3) = 1; //x translation to compare with other mls
    pcl::transformPointCloud(*cloud_smoothed2, *cloud_smoothed2_translated, cloud_translation);

    //    pclViewer4->addPointCloud (cloud, ColorHandlerRGB(cloud), "cloud");
    //    pclViewer4->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

    //    pclViewer4->addPointCloud (cloud_smoothed2_translated, ColorHandlerT(cloud_smoothed2_translated, 255.0, 0.0, 0.0), "smoothed2");
    //    pclViewer4->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "smoothed2");


    return cloud_smoothed2;
}

pcl::PointCloud<PointT>::Ptr euclideanClusters(pcl::PointCloud<PointT>::Ptr cloud, double distance){

    pcl::ScopeTime t("Euclidean Clusters");

    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance (distance);
    ec.setMinClusterSize (20);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud);
    ec.extract (cluster_indices);

    // pclViewer5->removeAllPointClouds();
    // pclViewer->addPointCloud (cloud, ColorHandlerT(cloud, 0.0, 255.0, 0.0), "scene_filtered");
    // pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "scene_filtered");

    //  std::cout << "Number of clusters : " << cluster_indices.at(0) << std::endl;

    pcl::PointCloud<PointT>::Ptr return_cloud(new pcl::PointCloud<PointT>);


    for (int i=0; i < cluster_indices.size(); i++){
        pcl::PointIndices cloud_indices = cluster_indices.at(i);
        pcl::PointCloud<PointT>::Ptr cloud_cluster(new pcl::PointCloud<PointT>);

        for (int j=0; j<cloud_indices.indices.size(); j++){
            cloud_cluster->push_back (cloud->points[cloud_indices.indices[j]]);
        }
        cloud_cluster->height = 1;
        cloud_cluster->width = cloud_cluster->size();

        *return_cloud += *cloud_cluster;

        segmented_clouds.push_back(cloud_cluster);

        pcl::visualization::PointCloudColorHandlerRandom<PointT> randColor(cloud_cluster);
        std::stringstream ss;
        ss << i;
        std::string ind = ss.str();
        std::string pc_name = "object_" + ind;
        pclViewer5->addPointCloud<PointT>(cloud_cluster, randColor, pc_name);
        pclViewer5->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, pc_name);

    }

    return return_cloud;
}


pcl::PointCloud<PointT>::Ptr segmentPointsBelongingToModel(pcl::PointCloud<PointT>::Ptr remaining_points, pcl::PointCloud<PointT>::Ptr model, double distance_threshold){

    pcl::ScopeTime t("Eliminating points near model");

    pcl::PointCloud<PointT>::Ptr segmented_cloud(new pcl::PointCloud<PointT>);

    double threshold_squared = distance_threshold * distance_threshold;


    pcl::KdTreeFLANN<PointT>::Ptr kdtree (new pcl::KdTreeFLANN<PointT>);
    kdtree->setInputCloud(model);
    std::vector<int> index;
    std::vector<float> sqr_distance;

    for(int i=0; i < remaining_points->size(); i++){
        kdtree->nearestKSearch(remaining_points->at(i), 1, index, sqr_distance);
        if(sqr_distance.at(0) >= threshold_squared){
            segmented_cloud->push_back(remaining_points->at(i));
        }
        //        cout << "neighbors: " << neighbors << endl;
        //        cout << "index: " << index.at(0) << endl;
        //        cout << "sqr_distance: " << sqr_distance.at(0) << endl;
    }


    // pclViewer5->addPointCloud (segmented_cloud, ColorHandlerT(segmented_cloud, 255.0, 255.0, 0.0), "segmented");
    // pclViewer5->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3 , "segmented");

    return segmented_cloud;
}

void region_growing_rgb(pcl::PointCloud<PointT>::Ptr cloud){
    pcl::ScopeTime t("region_growing_rgb");

    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    pcl::RegionGrowingRGB<PointT> reg;
    reg.setInputCloud (cloud);
    reg.setSearchMethod (tree);
    reg.setDistanceThreshold (_distanceThreshold);
    reg.setPointColorThreshold (_pointColorThreshold);
    reg.setRegionColorThreshold (_regionColorThreshold);
    reg.setMinClusterSize (_minClusterSize);

    std::vector <pcl::PointIndices> clusters;
    reg.extract (clusters);

    pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud <pcl::PointXYZRGB>);
    colored_cloud = reg.getColoredCloud ();

    pclViewer4->removeAllPointClouds();

    if(colored_cloud){
        std::cout << "Size = " << colored_cloud->size() << std::endl;
        pclViewer4->addPointCloud (colored_cloud, ColorHandlerRGB(colored_cloud), "segmentation");
        pclViewer4->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "segmentation");

    }
    //return colored_cloud;
}

void region_growing_rgb_thread(pcl::PointCloud<PointT>::Ptr cloud){

    region_growing_rgb(cloud);

    bool continueLoop = true;

    stringstream ss;
    ss.str(""); ss.clear();
    ss << _distanceThreshold;
    std::string distance_threshold_str = std::string(ss.str());
    ss.str(""); ss.clear();
    ss << _pointColorThreshold;
    std::string pointColorThreshold_str = std::string(ss.str());
    ss.str(""); ss.clear();
    ss << _regionColorThreshold;
    std::string regionColorThreshold_str = std::string(ss.str());
    ss.str(""); ss.clear();
    ss << _minClusterSize;
    std::string minClusterSize_str = std::string(ss.str());

    while(continueLoop){

        //std::cout << continueLoop  << std::endl;

        std::cout << "Choose one of the parameters to modify and press enter : "    << std::endl;
        std::cout << "1) Distance Threshold (" + distance_threshold_str + ")"       << std::endl;
        std::cout << "2) Point Color Threshold (" + pointColorThreshold_str + ")"   << std::endl;
        std::cout << "3) Region Color Threshold (" + regionColorThreshold_str + ")" << std::endl;
        std::cout << "4) Minimum Cluster Size (" + minClusterSize_str + ")"         << std::endl;
        std::cout << "5) Quit"                                                      << std::endl;

        string selection = "";
        std::cin >> selection;

        if(selection == "5") {
            break;
        }

        std::cout << "Enter the desired value : " << std::endl;
        double parameter_value = 0.0;
        std::string parameter_value_str = "";
        std::cin >> parameter_value_str;
        parameter_value = atof(parameter_value_str.c_str());

        if(selection == "1") {
            _distanceThreshold = parameter_value;
            std::cout << "Value set!" << std::endl;
            ss.str(""); ss.clear();
            ss << _distanceThreshold;
            distance_threshold_str = std::string(ss.str());
        }
        else if(selection == "2"){
            _pointColorThreshold = parameter_value;
            std::cout << "Value set!" << std::endl;
            ss.str(""); ss.clear();
            ss << _pointColorThreshold;
            pointColorThreshold_str = std::string(ss.str());
        }
        else if(selection == "3"){
            _regionColorThreshold = parameter_value;
            std::cout << "Value set!" << std::endl;
            ss.str(""); ss.clear();
            ss << _regionColorThreshold;
            regionColorThreshold_str = std::string(ss.str());
        }
        else if(selection == "4"){
            _minClusterSize = parameter_value;
            std::cout << "Value set!" << std::endl;
            ss.str(""); ss.clear();
            ss << _minClusterSize;
            minClusterSize_str = std::string(ss.str());

        }
        else{
            continueLoop = false;
        }

        region_growing_rgb(cloud);

    }

}



int main (int argc, char** argv){


    pcl::PointCloud<PointT>::Ptr in1(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr container_model(new pcl::PointCloud<PointT>);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr in1_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr container_model_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::io::loadPCDFile("../bmw_clutter.pcd", *in1_xyzrgb);
    pcl::io::loadPCDFile("../bmw_container_segmented.pcd", *container_model_xyzrgb);

    initPCLViewer();


    // Sampling
    in1_xyzrgb = downsample(in1_xyzrgb, 0.01);
    container_model_xyzrgb = downsample(container_model_xyzrgb, 0.005);


    // Compute normals
    in1 = computeSurfaceNormals(in1_xyzrgb);
    container_model = computeSurfaceNormals(container_model_xyzrgb);

    // Threads for computing the alignment with different techniques
    boost::thread thread_ISS(ISS_thread, in1, container_model);
    boost::thread thread_Sampling(Sampling_thread, in1, container_model);

    // Wait for threads to finish
    thread_ISS.join();
    thread_Sampling.join();

    // Check the best alignment and show it in PCL Viewer
    int bestTFIndex = getBestTransformIndex(&_AlignmentResults);
    //std::cout << "alignment score = " << _AlignmentResults.at(0)->fine_transformation_score << std::endl;
    showTransform(&_AlignmentResults, bestTFIndex);

    //saveClouds(&_AlignmentResults, bestTFIndex);
    CorrespondenceResults *best = _AlignmentResults.at(bestTFIndex);

    // Find Convex Hull of aligned model, shrink it and crop the points from the scene
    // belonging to the shrinked hull
    pcl::PointCloud<PointT>::Ptr scene_segmented(new pcl::PointCloud<PointT>);
    scene_segmented = cropAndSegmentScene(best->scene_cloud, best->model_fine_aligned, false, true);

    //    // Smooth remaining points
    //    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_segmented_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>);
    //    pcl::PointCloud<PointT>::Ptr smoothed_cloud(new pcl::PointCloud<PointT>);
    //    pcl::copyPointCloud(*scene_segmented, *scene_segmented_xyzrgb);
    //    smoothed_cloud = smoothPointCloud(scene_segmented_xyzrgb);

    // Find Clusters with euclidean clustering
    //    pcl::PointCloud<PointT>::Ptr euclidean_cloud = euclideanClusters(scene_segmented);
    pclViewer5->addPointCloud (best->scene_cloud, ColorHandlerT(best->scene_cloud, 255.0, 0.0, 0.0), "scene");
    pclViewer5->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "scene");
    //    pclViewer5->addPointCloud (euclidean_cloud, ColorHandlerT(euclidean_cloud, 255.0, 0.0, 255.0), "euclidean1");
    //    pclViewer5->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2 , "euclidean1");


    // Eliminate remaining points if they are near the aligned model
    pcl::PointCloud<PointT>::Ptr model_chopped = segmentPointsBelongingToModel(scene_segmented, best->model_fine_aligned, 0.01);


    // Do another round of euclidean clustering
    pcl::PointCloud<PointT>::Ptr euclidean_cloud = euclideanClusters(model_chopped, 0.015);
    // pclViewer5->addPointCloud (euclidean_cloud, ColorHandlerT(euclidean_cloud, 0.0, 255.0, 255.0), "euclidean");
    // pclViewer5->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3 , "euclidean");

  //  pcl::io::savePCDFileBinary("../bmw_clutter_remaining.pcd", *euclidean_cloud);

    //boost::thread regionThread(region_growing_rgb_thread, euclidean_cloud);


    while (!pclViewer->wasStopped()) {
        pclViewer->spinOnce (100);
        pclViewer5->spinOnce(100);
    }
    return 0;
}

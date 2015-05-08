#include <stdlib.h>
#include <sstream>
#include <stdio.h>

// PCL specific includes
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
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
#include <pcl/filters/passthrough.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/convex_hull.h>

#include <boost/thread.hpp>
#include <pcl/PCLPointCloud2.h>

typedef pcl::PointXYZRGBNormal PointT;
typedef pcl::PointXYZRGB PointC;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointC> ColorHandlerC;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointT> ColorHandlerT;
using namespace std;
using namespace pcl;
using namespace Eigen;

boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
//boost::shared_ptr<pcl::visualization::PCLVisualizer> pclViewer2 (new pcl::visualization::PCLVisualizer ("3D Viewer"));
PointCloud<PointT>::Ptr merged_pointcloud(new PointCloud<PointT>);

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

}

void showTransform(std::vector<CorrespondenceResults*> *results, int index){

    CorrespondenceResults *corr = results->at(index);

    pclViewer->removeAllPointClouds();

    pclViewer->addPointCloud (corr->scene_cloud, ColorHandlerT(corr->scene_cloud, 255.0, 0.0, 0.0), "scene");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "scene");

    // pclViewer->addPointCloud (corr->model_cloud, ColorHandlerT(corr->model_cloud, 0.0, 255.0, 0.0), "model");
    // pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "model");

    // pclViewer->addPointCloud (corr->model_coarse_aligned, ColorHandlerT(corr->model_coarse_aligned, 0.0, 255.0, 0.0), "ransac");
    // pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "ransac");

    pclViewer->addPointCloud (corr->model_fine_aligned, ColorHandlerT(corr->model_fine_aligned, 0.0, 0.0, 255.0), "icp");
    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "icp");

    //    pclViewer->addPointCloud (corr->scene_keypoints, ColorHandlerT(corr->scene_keypoints, 255.0, 255.0, 0.0), "scene_key");
    //    pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_key");

    // pclViewer->addPointCloud (corr->model_keypoints, ColorHandlerT(corr->model_keypoints, 0.0, 0.0, 255.0), "model_key");
    // pclViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "model_key");

    //  pclViewer->addCorrespondences<PointT>(corr->model_keypoints, corr->scene_keypoints, corr->correspondences, "corr1");

}


void saveClouds(std::vector<CorrespondenceResults*> *results, int index){

    CorrespondenceResults *corr = results->at(index);

    pcl::io::savePCDFileASCII<PointT>(std::string("../customAlignment_scene.pcd"),  *corr->scene_cloud);
    pcl::io::savePCDFileASCII<PointT>(std::string("../customAlignment_coarse.pcd"), *corr->model_coarse_aligned);
    pcl::io::savePCDFileASCII<PointT>(std::string("../customAlignment_fine.pcd"),   *corr->model_fine_aligned);

}

PointCloud<PointT>::Ptr computeSurfaceNormals(PointCloud<PointT>::Ptr cloud)
{
    pcl::ScopeTime t("Normals");

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

    // Estimate the normals.
    pcl::NormalEstimationOMP<PointT, pcl::Normal> normalEstimation;
    PointCloud<pcl::Normal>::Ptr normals(new PointCloud<pcl::Normal>);
    search::KdTree<PointT>::Ptr kdtree(new search::KdTree<PointT>);

    normalEstimation.setInputCloud(cloud);
    //normalEstimation.setKSearch(10);
    normalEstimation.setRadiusSearch(0.025);
    normalEstimation.setSearchMethod(kdtree);
    normalEstimation.compute(*normals);

    PointCloud<PointT>::Ptr normal_cloud (new PointCloud<PointT>);
    pcl::concatenateFields(*cloud, *normals, *normal_cloud);

    return normal_cloud;
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
    corr_rejector.setInlierThreshold(0.05);
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


std::vector<PointCloud<PointT>::Ptr> parseFile(string filePath){

    std::vector<PointCloud<PointT>::Ptr> pc_vec;

    string line;
    ifstream myfile(filePath.c_str());
    if(myfile.is_open()){
        while ( getline (myfile,line) )
        {
            if(line.length() > 0){
                cout << line << '\n';
                PointCloud<PointT>::Ptr temp(new PointCloud<PointT>);
                string path = "../" + line;
                pcl::io::loadPCDFile(path, *temp);
                cout << "size : " << temp->size() << endl;
                pc_vec.push_back(temp);
            }
        }
        myfile.close();
    }
    return pc_vec;
}


pcl::PointCloud<PointT>::Ptr symmetry(pcl::PointCloud<PointT>::Ptr cloud){


}



int main (int argc, char** argv){


    PointCloud<PointT>::Ptr first(new PointCloud<PointT>);
    PointCloud<PointT>::Ptr second(new PointCloud<PointT>);

    std::vector<PointCloud<PointT>::Ptr> clouds = parseFile("../clouds.txt");

    if(clouds.size() > 1){

        // Iterate over all the clouds to merge
        for(int i = 1; i < clouds.size(); i++){

            // pcl::io::loadPCDFile("../scans/bin1.pcd", *first);
            // pcl::io::loadPCDFile("../scans/bin2.pcd", *second);

            if(merged_pointcloud->size() <= 0){
                merged_pointcloud = clouds.at(0);
            }

            second = clouds.at(i);

            // Downsample
            PointCloud<PointT>::Ptr downsampled_merged(new PointCloud<PointT>);
            PointCloud<PointT>::Ptr downsampled_second(new PointCloud<PointT>);
            downsampled_merged  = downsample(merged_pointcloud , 0.005);
            downsampled_second = downsample(second, 0.005);

            /// COMMENT IF USING BINS ONLY
            //            pcl::PointCloud<PointT>::Ptr temp_cloud(new pcl::PointCloud<PointT>);
            //            pcl::PassThrough<PointT> pass_filter;
            //            pass_filter.setFilterFieldName("z");
            //            pass_filter.setFilterLimits(0, 1.3);
            //            pass_filter.setInputCloud(downsampled_merged);
            //            pass_filter.filter(*temp_cloud);
            //            *downsampled_merged = *temp_cloud;
            //            pass_filter.setInputCloud(downsampled_second);
            //            pass_filter.filter(*temp_cloud);
            //            *downsampled_second = *temp_cloud;
            /// END COMMENT

            // Normals
            downsampled_merged  = computeSurfaceNormals(downsampled_merged);
            downsampled_second = computeSurfaceNormals(downsampled_second);

            initPCLViewer();

            // Threads for computing the alignment with different techniques
            boost::thread thread_ISS(ISS_thread, downsampled_merged, downsampled_second);
            boost::thread thread_Sampling(Sampling_thread, downsampled_merged, downsampled_second);

            // Wait for threads to finish
            thread_ISS.join();
            thread_Sampling.join();

            // Check the best alignment and show it in PCL Viewer
            int bestTFIndex = getBestTransformIndex(&_AlignmentResults);
            //std::cout << "alignment score = " << _AlignmentResults.at(0)->fine_transformation_score << std::endl;
            showTransform(&_AlignmentResults, bestTFIndex);

            //saveClouds(&_AlignmentResults, bestTFIndex);

            CorrespondenceResults *corr = _AlignmentResults.at(bestTFIndex);
            *merged_pointcloud += *(corr->model_fine_aligned);
            _AlignmentResults.clear();

            pclViewer->spinOnce (100);
        }
    }

    merged_pointcloud = downsample(merged_pointcloud , 0.005);
    merged_pointcloud = computeSurfaceNormals(merged_pointcloud);
    pcl::io::savePCDFile("../merged.pcd", *merged_pointcloud);





    while (!pclViewer->wasStopped()) {
        pclViewer->spinOnce (100);
    }
    return 0;
}

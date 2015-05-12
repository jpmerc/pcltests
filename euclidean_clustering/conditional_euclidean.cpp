#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/filters/passthrough.h>

typedef pcl::PointXYZRGBNormal PointT;

pcl::PointCloud<PointT>::Ptr computeSurfaceNormals(pcl::PointCloud<PointT>::Ptr cloud)
{

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

pcl::PointCloud<PointT>::Ptr extractPlane(pcl::PointCloud<PointT>::Ptr cloud, bool removePlane){

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
    extract.setNegative(removePlane);
    extract.setInputCloud(cloud);
    extract.setIndices(inlierIndices);
    extract.filter (*returned_cloud);

    return returned_cloud;

}

bool customClustering (const PointT& point_a, const PointT& point_b, float squared_distance)
{
    using namespace Eigen;

    Vector3f point_a_normal = point_a.getNormalVector3fMap();
    Vector3f point_b_normal = point_b.getNormalVector3fMap();

    Vector3i point_a_color_i = point_a.getRGBVector3i();
    Vector3i point_b_color_i = point_b.getRGBVector3i();

    Vector3f point_a_color( (point_a_color_i(0) / 255.0) , (point_a_color_i(1) / 255.0) , (point_a_color_i(2) / 255.0));
    Vector3f point_b_color( (point_b_color_i(0) / 255.0) , (point_b_color_i(1) / 255.0) , (point_b_color_i(2) / 255.0));

    double point_a_curvature = point_a.curvature;
    double point_b_curvature = point_b.curvature;


    double normal_score = 1 - std::abs(point_a_normal.dot(point_b_normal)) ;
    double spatial_score = sqrt(squared_distance);
    double color_score = ( (point_a_color - point_b_color).norm() ) / 3;
    double curvature_score = std::abs(point_a_curvature - point_b_curvature);



//    if(normal_score < 0.01){
//        return true;
//    }

    if(normal_score < 0.01){

        if(curvature_score < 0.01){
            return true;
        }
        else return false;
    }


    else{
        std::cout << "n=" << normal_score << "  curv=" <<  curvature_score << "  col=" << color_score << "  d=" << spatial_score << std::endl;
        return false;
    }

}

int
main (int argc, char** argv)
{
  // Read in the cloud data
  pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>), cloud_f (new pcl::PointCloud<PointT>);
    if (pcl::io::loadPCDFile<PointT>(argv[1], *cloud) != 0)
    {
        return -1;
    }

  std::cout << "PointCloud before filtering has: " << cloud->points.size () << " data points." << std::endl; //*

  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<PointT> vg;
  pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
  vg.setInputCloud (cloud);
  vg.setLeafSize (0.005f, 0.005f, 0.005f);
  vg.filter (*cloud_filtered);
  std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl; //*

  cloud_filtered = computeSurfaceNormals(cloud_filtered);


  /// COMMENT IF NOT USING OSD DATASET
    pcl::PointCloud<PointT>::Ptr temp_cloud(new pcl::PointCloud<PointT>);
    pcl::PassThrough<PointT> pass_filter;
    pass_filter.setFilterFieldName("z");
    pass_filter.setFilterLimits(0, 1.3);
    pass_filter.setInputCloud(cloud_filtered);
    pass_filter.filter(*temp_cloud);
    cloud_filtered = extractPlane(temp_cloud, true);

  // Create the segmentation object for the planar model and set all the parameters
//  pcl::SACSegmentation<PointT> seg;
//  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
//  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
//  pcl::PointCloud<PointT>::Ptr cloud_plane (new pcl::PointCloud<PointT> ());
  pcl::PCDWriter writer;
//  seg.setOptimizeCoefficients (true);
//  seg.setModelType (pcl::SACMODEL_PLANE);
//  seg.setMethodType (pcl::SAC_RANSAC);
//  seg.setMaxIterations (100);
//  seg.setDistanceThreshold (0.02);

//  int i=0, nr_points = (int) cloud_filtered->points.size ();
//  while (cloud_filtered->points.size () > 0.3 * nr_points)
//  {
//    // Segment the largest planar component from the remaining cloud
//    seg.setInputCloud (cloud_filtered);
//    seg.segment (*inliers, *coefficients);
//    if (inliers->indices.size () == 0)
//    {
//      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
//      break;
//    }

//    // Extract the planar inliers from the input cloud
//    pcl::ExtractIndices<PointT> extract;
//    extract.setInputCloud (cloud_filtered);
//    extract.setIndices (inliers);
//    extract.setNegative (false);

//    // Get the points associated with the planar surface
//    extract.filter (*cloud_plane);
//    std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

//    // Remove the planar inliers, extract the rest
//    extract.setNegative (true);
//    extract.filter (*cloud_f);
//    *cloud_filtered = *cloud_f;
//  }

  // Creating the KdTree object for the search method of the extraction
//  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
//  tree->setInputCloud (cloud_filtered);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::ConditionalEuclideanClustering<PointT> cec (true);
  cec.setInputCloud (cloud_filtered);
  cec.setConditionFunction (&customClustering);
  cec.setClusterTolerance (0.02);
  cec.setMinClusterSize (20);
  cec.setMaxClusterSize (250000);
  cec.segment (cluster_indices);

//  std::vector<pcl::PointIndices> cluster_indices;
//  pcl::EuclideanClusterExtraction<PointT> ec;
//  ec.setClusterTolerance (0.02); // 2cm
//  ec.setMinClusterSize (300);
//  ec.setMaxClusterSize (25000);
//  ec.setSearchMethod (tree);
//  ec.setInputCloud (cloud_filtered);
//  ec.extract (cluster_indices);

  int j = 0;
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    pcl::PointCloud<PointT>::Ptr cloud_cluster (new pcl::PointCloud<PointT>);
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
      cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*
    cloud_cluster->width = cloud_cluster->points.size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
    std::stringstream ss;
    ss << "cloud_cluster_" << j << ".pcd";
    writer.write<PointT> (ss.str (), *cloud_cluster, false); //*
    j++;
  }

  return (0);
}

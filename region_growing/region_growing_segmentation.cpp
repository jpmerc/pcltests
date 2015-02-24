#include <iostream>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/voxel_grid.h>

pcl::PointCloud<pcl::PointXYZ>::Ptr smoothSurfaces(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_to_smooth){

    // Surface Approximation
    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_with_approximated_surface(new pcl::PointCloud<pcl::PointXYZ>);
    mls.setInputCloud (cloud_to_smooth);
    mls.setSearchRadius (0.04);
    mls.setPolynomialFit (true);
    mls.setPolynomialOrder (2);
    mls.process (*cloud_with_approximated_surface);


    // Upsampling
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_upsampled(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls2;
    mls2.setInputCloud (cloud_with_approximated_surface);
    mls2.setSearchRadius (0.02);
    mls2.setPolynomialFit (true);
    mls2.setPolynomialOrder (2);
    mls2.setUpsamplingMethod (pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ>::SAMPLE_LOCAL_PLANE);
    mls2.setUpsamplingRadius (0.01);
    mls2.setUpsamplingStepSize (0.005);
    mls2.process (*cloud_upsampled);

    // filter the upsampled pointcloud to get the same point density as all other pointclouds
    pcl::VoxelGrid<pcl::PointXYZ> vx_grid;
    vx_grid.setInputCloud (cloud_upsampled);
    double leaf_size = 0.005;
    vx_grid.setLeafSize (leaf_size, leaf_size, leaf_size);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    vx_grid.filter (*cloud_filtered);

    return cloud_filtered;
}

int
main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile(argv[1], *cloud);
  
  std::vector<int> indices_nan;
  pcl::removeNaNFromPointCloud(*cloud, *cloud, indices_nan);
  
  //cloud = smoothSurfaces(cloud);


  pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod (tree);
  normal_estimator.setInputCloud (cloud);
  normal_estimator.setKSearch (50);
  normal_estimator.compute (*normals);

  pcl::IndicesPtr indices (new std::vector <int>);
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.0, 1.0);
  pass.filter (*indices);

  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  reg.setMinClusterSize (50);
  reg.setMaxClusterSize (1000000);
  reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (30);
  reg.setInputCloud (cloud);
  //reg.setIndices (indices);
  reg.setInputNormals (normals);
  reg.setSmoothnessThreshold (10.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold (1.0);

  std::vector <pcl::PointIndices> clusters;
  reg.extract (clusters);

  std::cout << "Number of clusters is equal to " << clusters.size () << std::endl;
  std::cout << "First cluster has " << clusters[0].indices.size () << " points." << endl;
  std::cout << "These are the indices of the points of the initial" <<
    std::endl << "cloud that belong to the first cluster:" << std::endl;
  int counter = 0;
  while (counter < clusters[0].indices.size ())
  {
    std::cout << clusters[0].indices[counter] << ", ";
    counter++;
    if (counter % 10 == 0)
      std::cout << std::endl;
  }
  std::cout << std::endl;

  pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
  pcl::visualization::CloudViewer viewer ("Cluster viewer");
  viewer.showCloud(colored_cloud);
  while (!viewer.wasStopped ())
  {
  }

  return (0);
}

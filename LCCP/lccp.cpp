#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/common/time.h>
#include <pcl/common/transforms.h>

#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/lccp_segmentation.h>

#include <vtkImageReader2Factory.h>
#include <vtkImageReader2.h>
#include <vtkImageData.h>
#include <vtkImageFlip.h>
#include <vtkPolyLine.h>

#include <boost/format.hpp>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::visualization::PointCloudColorHandlerRGBField<PointT> ColorHandlerRGB;

typedef pcl::LCCPSegmentation<PointT>::SupervoxelAdjacencyList SuperVoxelAdjacencyList;

bool show_normals = false;
bool normals_changed = false;
bool show_adjacency = false;
bool show_supervoxels = false;
bool show_visualization = true;
bool show_help = true;
float normals_scale;

///  Default values of parameters before parsing
// Supervoxel Stuff
float voxel_resolution = 0.0075f;
float seed_resolution = 0.03f;
float color_importance = 0.0f;
float spatial_importance = 1.0f;
float normal_importance = 4.0f;
bool use_single_cam_transform = false;
bool use_supervoxel_refinement = false;

// LCCPSegmentation Stuff
float concavity_tolerance_threshold = 10;
float smoothness_threshold = 0.1;
uint32_t min_segment_size = 0;
bool use_extended_convexity = false;
bool use_sanity_criterion = false;

void
keyboardEventOccurred (const pcl::visualization::KeyboardEvent& event_arg,
                       void*)
{
  int key = event_arg.getKeyCode ();

  if (event_arg.keyUp ())
    switch (key)
    {
      case (int) '1':
        show_normals = !show_normals;
        normals_changed = true;
        break;
      case (int) '2':
        show_adjacency = !show_adjacency;
        break;
      case (int) '3':
        show_supervoxels = !show_supervoxels;
        break;
      case (int) '4':
        normals_scale *= 1.25;
        normals_changed = true;
        break;
      case (int) '5':
        normals_scale *= 0.8;
        normals_changed = true;
        break;
      case (int) 'd':
      case (int) 'D':
        show_help = !show_help;
        break;
      default:
        break;
    }
}

void
printText (boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_arg)
{
    std::string on_str = "ON";
    std::string off_str = "OFF";
    if (!viewer_arg->updateText ("Press (1-n) to show different elements (d) to disable this", 5, 72, 12, 1.0, 1.0, 1.0, "hud_text"))
        viewer_arg->addText ("Press (1-n) to show different elements", 5, 72, 12, 1.0, 1.0, 1.0, "hud_text");

    std::string temp = "(1) Supervoxel Normals, currently " + ( (show_normals) ? on_str : off_str);
    if (!viewer_arg->updateText (temp, 5, 60, 10, 1.0, 1.0, 1.0, "normals_text"))
        viewer_arg->addText (temp, 5, 60, 10, 1.0, 1.0, 1.0, "normals_text");

    temp = "(2) Adjacency Graph, currently " + ( (show_adjacency) ? on_str : off_str) + "\n      White: convex; Red: concave";
    if (!viewer_arg->updateText (temp, 5, 38, 10, 1.0, 1.0, 1.0, "graph_text"))
        viewer_arg->addText (temp, 5, 38, 10, 1.0, 1.0, 1.0, "graph_text");

    temp = "(3) Press to show " + ( (show_supervoxels) ? std::string ("SEGMENTATION") : std::string ("SUPERVOXELS"));
    if (!viewer_arg->updateText (temp, 5, 26, 10, 1.0, 1.0, 1.0, "supervoxel_text"))
        viewer_arg->addText (temp, 5, 26, 10, 1.0, 1.0, 1.0, "supervoxel_text");

    temp = "(4/5) Press to increase/decrease normals scale, currently " + boost::str (boost::format ("%.3f") % normals_scale);
    if (!viewer_arg->updateText (temp, 5, 14, 10, 1.0, 1.0, 1.0, "normals_scale_text"))
        viewer_arg->addText (temp, 5, 14, 10, 1.0, 1.0, 1.0, "normals_scale_text");
}

void
removeText (boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_arg)
{
    viewer_arg->removeShape ("hud_text");
    viewer_arg->removeShape ("normals_text");
    viewer_arg->removeShape ("graph_text");
    viewer_arg->removeShape ("supervoxel_text");
    viewer_arg->removeShape ("normals_scale_text");
}



int main (int argc, char** argv)
{
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>),
            cloud_filtered (new pcl::PointCloud<PointT>),
            cloud_filtered_translated (new pcl::PointCloud<PointT>);


    pcl::io::loadPCDFile("../bmw_color_balls.pcd", *cloud);

    // Params
    normals_scale = seed_resolution / 2.0;
    uint k_factor = 0;
    if (use_extended_convexity) k_factor = 1;

    /// Supervoxels
    pcl::SupervoxelClustering<PointT> super (voxel_resolution, seed_resolution);
    super.setUseSingleCameraTransform (use_single_cam_transform);
    super.setInputCloud (cloud);
    super.setColorImportance (color_importance);
    super.setSpatialImportance (spatial_importance);
    super.setNormalImportance (normal_importance);
    std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters;

    PCL_INFO ("Extracting supervoxels\n");
    super.extract (supervoxel_clusters);

    if (use_supervoxel_refinement)
    {
        PCL_INFO ("Refining supervoxels\n");
        super.refineSupervoxels (2, supervoxel_clusters);
    }
    std::stringstream temp;
    temp << "  Nr. Supervoxels: " << supervoxel_clusters.size () << "\n";
    PCL_INFO (temp.str ().c_str ());

    PCL_INFO ("Getting supervoxel adjacency\n");
    std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
    super.getSupervoxelAdjacency (supervoxel_adjacency);


    /// Get the cloud of supervoxel centroid with normals and the colored cloud with supervoxel coloring (this is used for visulization)
    pcl::PointCloud<pcl::PointNormal>::Ptr sv_centroid_normal_cloud = pcl::SupervoxelClustering<PointT>::makeSupervoxelNormalCloud (supervoxel_clusters);


    /// The Main Step: Perform LCCPSegmentation

    PCL_INFO ("Starting Segmentation\n");
    pcl::LCCPSegmentation<PointT> lccp;
    lccp.setConcavityToleranceThreshold (concavity_tolerance_threshold);
    lccp.setSanityCheck (use_sanity_criterion);
    lccp.setSmoothnessCheck (true, voxel_resolution, seed_resolution, smoothness_threshold);
    lccp.setKFactor (k_factor);
    lccp.segment (supervoxel_clusters, supervoxel_adjacency);

    if (min_segment_size > 0)
    {
        PCL_INFO ("Merging small segments\n");
        lccp.mergeSmallSegments (min_segment_size);
    }

    PCL_INFO ("Interpolation voxel cloud -> input cloud and relabeling\n");
    pcl::PointCloud<pcl::PointXYZL>::Ptr sv_labeled_cloud = super.getLabeledCloud ();
    pcl::PointCloud<pcl::PointXYZL>::Ptr lccp_labeled_cloud = sv_labeled_cloud->makeShared ();
    lccp.relabelCloud (*lccp_labeled_cloud);
    SuperVoxelAdjacencyList sv_adjacency_list;
    lccp.getSVAdjacencyList (sv_adjacency_list);  // Needed for visualization



    if (show_visualization)
    {
        /// Calculate visualization of adjacency graph
        // Using lines this would be VERY slow right now, because one actor is created for every line (may be fixed in future versions of PCL)
        // Currently this is a work-around creating a polygon mesh consisting of two triangles for each edge
        using namespace pcl;

        typedef LCCPSegmentation<PointT>::VertexIterator VertexIterator;
        typedef LCCPSegmentation<PointT>::AdjacencyIterator AdjacencyIterator;
        typedef LCCPSegmentation<PointT>::EdgeID EdgeID;

        std::set<EdgeID> edge_drawn;

        const unsigned char convex_color [3] = {255, 255, 255};
        const unsigned char concave_color [3] = {255, 0, 0};
        const unsigned char* color;

        //The vertices in the supervoxel adjacency list are the supervoxel centroids
        //This iterates through them, finding the edges
        std::pair<VertexIterator, VertexIterator> vertex_iterator_range;
        vertex_iterator_range = boost::vertices (sv_adjacency_list);

        /// Create a cloud of the voxelcenters and map: VertexID in adjacency graph -> Point index in cloud

        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New ();
        vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New ();
        vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New ();
        colors->SetNumberOfComponents (3);
        colors->SetName ("Colors");

        // Create a polydata to store everything in
        vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New ();
        for (VertexIterator itr = vertex_iterator_range.first; itr != vertex_iterator_range.second; ++itr)
        {
            const uint32_t sv_label = sv_adjacency_list[*itr];
            std::pair<AdjacencyIterator, AdjacencyIterator> neighbors = boost::adjacent_vertices (*itr, sv_adjacency_list);

            for (AdjacencyIterator itr_neighbor = neighbors.first; itr_neighbor != neighbors.second; ++itr_neighbor)
            {
                EdgeID connecting_edge = boost::edge (*itr, *itr_neighbor, sv_adjacency_list).first;  //Get the edge connecting these supervoxels
                if (sv_adjacency_list[connecting_edge].is_convex)
                    color = convex_color;
                else
                    color = concave_color;

                // two times since we add also two points per edge
                colors->InsertNextTupleValue (color);
                colors->InsertNextTupleValue (color);

                pcl::Supervoxel<PointT>::Ptr supervoxel = supervoxel_clusters.at (sv_label);
                pcl::PointXYZRGBA vert_curr = supervoxel->centroid_;


                const uint32_t sv_neighbor_label = sv_adjacency_list[*itr_neighbor];
                pcl::Supervoxel<PointT>::Ptr supervoxel_neigh = supervoxel_clusters.at (sv_neighbor_label);
                pcl::PointXYZRGBA vert_neigh = supervoxel_neigh->centroid_;

                points->InsertNextPoint (vert_curr.data);
                points->InsertNextPoint (vert_neigh.data);

                // Add the points to the dataset
                vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New ();
                polyLine->GetPointIds ()->SetNumberOfIds (2);
                polyLine->GetPointIds ()->SetId (0, points->GetNumberOfPoints ()-2);
                polyLine->GetPointIds ()->SetId (1, points->GetNumberOfPoints ()-1);
                cells->InsertNextCell (polyLine);
            }
        }
        polyData->SetPoints (points);
        // Add the lines to the dataset
        polyData->SetLines (cells);

        polyData->GetPointData ()->SetScalars (colors);

        /// END: Calculate visualization of adjacency graph

        /// Configure Visualizer
        pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        viewer->setBackgroundColor (0, 0, 0);
        viewer->registerKeyboardCallback (keyboardEventOccurred, 0);
        viewer->addPointCloud (lccp_labeled_cloud, "maincloud");

        /// Visualization Loop
        PCL_INFO ("Loading viewer\n");
        while (!viewer->wasStopped ())
        {
            viewer->spinOnce (100);

            /// Show Segmentation or Supervoxels
            viewer->updatePointCloud ( (show_supervoxels) ? sv_labeled_cloud : lccp_labeled_cloud, "maincloud");

            /// Show Normals
            if (normals_changed)
            {
                viewer->removePointCloud ("normals");
                if (show_normals)
                {
                    viewer->addPointCloudNormals<pcl::PointNormal> (sv_centroid_normal_cloud, 1, normals_scale, "normals");
                    normals_changed = false;
                }
            }
            /// Show Adjacency
            if (show_adjacency)
            {
                viewer->removeShape ("adjacency_graph");
                viewer->addModelFromPolyData (polyData, "adjacency_graph");
            }
            else
            {
                viewer->removeShape ("adjacency_graph");
            }

            if (show_help)
            {
                viewer->removeShape ("help_text");
                printText (viewer);
            }
            else
            {
                removeText (viewer);
                if (!viewer->updateText ("Press d to show help", 5, 10, 12, 1.0, 1.0, 1.0, "help_text"))
                    viewer->addText ("Press d to show help", 5, 10, 12, 1.0, 1.0, 1.0, "help_text");
            }

            boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        }
    }  /// END if (show_visualization)






    return (0);
}

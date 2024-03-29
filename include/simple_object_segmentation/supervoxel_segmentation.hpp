// Copyright (C) 2016 by Krishneel Chaudhary @ JSK Lab,
// The University of Tokyo, Japan
// krishneel@jsk.imi.i.u-tokyo.ac.jp

#ifndef _SUPERVOXEL_SEGMENTATION_H_
#define _SUPERVOXEL_SEGMENTATION_H_

#include <ros/ros.h>
#include <ros/console.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>
#include <dynamic_reconfigure/server.h>
#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <simple_object_segmentation/SupervoxelSegmentationConfig.h>

namespace sos = simple_object_segmentation;

class SupervoxelSegmentation {
 protected:
    double color_importance_;
    double spatial_importance_;
    double normal_importance_;
    bool use_transform_;
    boost::mutex mutex_;

 private:
    typedef sos::SupervoxelSegmentationConfig Config;
    virtual void configCallback(Config &, uint32_t);
    boost::shared_ptr<dynamic_reconfigure::Server<Config> > srv_;

 public:
    SupervoxelSegmentation();
    typedef pcl::PointXYZRGB PointT;
    typedef boost::shared_ptr<SupervoxelSegmentation> Ptr;

    void supervoxelSegmentation(
       const pcl::PointCloud<PointT>::Ptr,
       std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr > &,
       pcl::SupervoxelClustering<PointT>::VoxelAdjacencyList &);
    void supervoxelSegmentation(
       const pcl::PointCloud<PointT>::Ptr,
       std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr > &,
       std::multimap<uint32_t, uint32_t> &);
    pcl::PointCloud<PointT>::Ptr publishSupervoxel(
       const std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr>,
       sensor_msgs::PointCloud2 &,
       jsk_recognition_msgs::ClusterPointIndices &,
       const std_msgs::Header &);
    std::vector<pcl_msgs::PointIndices> convertToROSPointIndices(
       const std::vector<pcl::PointIndices>,
       const std_msgs::Header &);
    void sortSupervoxelsByCentroid(
       std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr > &,
       std::multimap<uint32_t, uint32_t> &);

    double coplanar_threshold_;
    double seed_resolution_;
    double distance_threshold_;
    double angle_threshold_;
    double voxel_resolution_;

    typedef pcl::SupervoxelClustering<PointT>::VoxelAdjacencyList AdjacencyList;
    typedef std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> SupervoxelMap;
    typedef std::multimap<uint32_t, uint32_t> AdjacentList;
};


#endif  //_SUPERVOXEL_SEGMENTATION_H_

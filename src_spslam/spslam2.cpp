//--------------------------------C-Code

#include "./include/System.h"
#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>
#include <opencv2/core/core.hpp>

#include <string>
#include "lib/conversion.h"

namespace bpy = boost::python;
using namespace std;

//  need to initialize numpy
#if PY_MAJOR_VERSION >= 3
int
#else
void
#endif
init_numpy()
{
    bpy::numeric::array::set_module_and_type("numpy", "ndarray");
    import_array();
}


class SLAMClass{
    public:
        SLAMClass();

        NDArrayConverter cvt;

        void init(const std::string &path_to_vocabulary, const std::string &path_to_settings, const string camera_type, const bool flag_visualize){
            ORB_SLAM2::Config::SetParameterFile(path_to_settings);
            this->slam_obj = new ORB_SLAM2::System(path_to_vocabulary, path_to_settings, ORB_SLAM2::System::RGBD, flag_visualize);
            std::cout << path_to_vocabulary << std::endl;
        }

        PyObject* track_rgbd(PyObject* rgb_img, PyObject* d_img, const double timestamp){
            cv::Mat rgb_img_mat = cvt.toMat(rgb_img);
            cv::Mat d_img_mat = cvt.toMat(d_img);
//            d_img_mat.convertTo(d_img_mat, CV_32F);
            auto camera_matrix = this->slam_obj->TrackRGBD(rgb_img_mat, d_img_mat, timestamp);
            PyObject* ret = cvt.toNDArray(camera_matrix);
            return ret;
        }

        void prepare_dump(){
            slam_obj->PrepareDump();
        }

            // stuff for export
//    cv::Mat kf_ids_from_mps;
//    cv::Mat kf_ids;
//    cv::Mat plane_ids;
//    cv::Mat mp_3dpts;
//    cv::Mat kf_3dpts;
//    cv::Mat plane_ids_from_boundary_pts;
//    cv::Mat plane_params;
//    cv::Mat plane_boundary_pts;

        PyObject* get_kf_ids_from_mps(){
            return cvt.toNDArray(slam_obj->dump_kf_ids_from_mps());
        }
        PyObject* get_kf_ids(){
            return cvt.toNDArray(slam_obj->dump_kf_ids());
        }
        PyObject* get_mp_3dpts(){
            return cvt.toNDArray(slam_obj->dump_mp_3dpts());
        }
        PyObject* get_kf_3dpts(){
            return cvt.toNDArray(slam_obj->dump_kf_3dpts());
        }
        PyObject* get_plane_params(){
            return cvt.toNDArray(slam_obj->dump_plane_params());
        }
        PyObject* get_plane_ids(){
            return cvt.toNDArray(slam_obj->dump_plane_ids());
        }
        PyObject* get_plane_ids_from_boundary_pts(){
            return cvt.toNDArray(slam_obj->dump_plane_ids_from_boundary_pts());
        }
        PyObject* get_boundary_pts(){
            return cvt.toNDArray(slam_obj->dump_boundary_pts());
        }

        PyObject* get_boundary_update_sizes(){
            return cvt.toNDArray(slam_obj->dump_cntBoundaryUpdateSizes());
        }

        PyObject* get_grid_map(){
            return cvt.toNDArray(slam_obj->get_gridmaps());
        }


        ~SLAMClass();


    private:
        ORB_SLAM2::System *slam_obj;
};

    SLAMClass::SLAMClass() {
        // nothing
    }

    SLAMClass::~SLAMClass() {
        delete this->slam_obj; // Otherwise may end up with a SegFault!!
    }

//--------------------------------Boost-Python-Code


BOOST_PYTHON_MODULE(ORBSLAM2)
{
    using namespace boost::python;
    Py_Initialize(); // required by NumPy
    init_numpy(); // required by NumPy

    class_<SLAMClass>("SLAM")
        .def("init", &SLAMClass::init)
        .def("track_rgbd", &SLAMClass::track_rgbd)
.def("prepare_dump", &SLAMClass::prepare_dump)
.def("get_kf_ids_from_mps", &SLAMClass::get_kf_ids_from_mps)
.def("get_kf_ids", &SLAMClass::get_kf_ids)
.def("get_plane_ids", &SLAMClass::get_plane_ids)
.def("get_mp_3dpts", &SLAMClass::get_mp_3dpts)
.def("get_kf_3dpts", &SLAMClass::get_kf_3dpts)
.def("get_plane_ids_from_boundary_pts", &SLAMClass::get_plane_ids_from_boundary_pts)
.def("get_plane_params", &SLAMClass::get_plane_params)
.def("get_boundary_pts", &SLAMClass::get_boundary_pts)
.def("get_boundary_update_sizes", &SLAMClass::get_boundary_update_sizes)
.def("get_grid_map", &SLAMClass::get_grid_map);
//    cv::Mat kf_ids_from_mps;
//    cv::Mat kf_ids;
//    cv::Mat plane_ids;
//    cv::Mat mp_3dpts
//    cv::Mat kf_3dpts;
//    cv::Mat plane_ids_from_boundary_pts;
//    cv::Mat plane_params;
//    cv::Mat plane_boundary_pts;

}

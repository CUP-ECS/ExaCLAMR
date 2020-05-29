#ifndef EXACLAMR_MESH_HPP
#define EXACLAMR_MESH_HPP

#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include<mpi.h>

#include <memory>

namespace ExaCLAMR
{
template<class MemorySpace>
class Mesh
{
    public:

        Mesh( const std::array<double, 6>& global_bounding_box, 
                const std::array<int, 3>& global_num_cell,
                const std::array<bool, 3>& periodic,
                const Cajita::Partitioner& partitioner, 
                const int halo_width, 
                MPI_Comm comm ) 
        {
            // 2-D Mesh for now
            std::array<int, 3> num_cell;
            num_cell[0] = global_num_cell[0];
            num_cell[1] = global_num_cell[1];
            num_cell[2] = 1;

            for (int dim = 0; dim < 3; ++dim ) {
                _min_domain_global_cell_index[dim] = 0;
                _max_domain_global_cell_index[dim] = num_cell[dim];
            }

            std::array<double, 6> bounding_box;
            bounding_box[0] = global_bounding_box[0];
            bounding_box[1] = global_bounding_box[1];
            bounding_box[2] = 0;
            bounding_box[3] = global_bounding_box[3];
            bounding_box[4] = global_bounding_box[4];
            bounding_box[5] = 1;

            std::array<double, 3> cell_size;
            cell_size[0] = ( bounding_box[3] - bounding_box[0] ) / ( double ) num_cell[0];
            cell_size[1] = ( bounding_box[4] - bounding_box[1] ) / ( double ) num_cell[1];
            cell_size[2] = ( bounding_box[5] - bounding_box[2] ) / ( double ) num_cell[2];

            /*
            for ( int dim = 0; dim < 3; ++dim ) {
                double extent = num_cell[dim] * cell_size[dim];
                if ( std::abs( extent - ( global_bounding_box[dim + 3] - global_bounding_box[dim] ) ) > double( 10.0 ) * std::numeric_limits<double>::epsilon() )
                    throw std::logic_error( "Extent not evenly divisible by uniform cell size" );
            }
            */
            /*
            printf("Num Cells: %d, %d, %d\n", num_cell[0], num_cell[1], num_cell[2]);
            printf("Bounding Box: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f\n", bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3], bounding_box[4], bounding_box[5]);
            printf("Cell Size: %.4f\n", cell_size);
            */

            std::array<double, 3> global_low_corner = { bounding_box[0], bounding_box[1], bounding_box[2] };
            std::array<double, 3> global_high_corner = { bounding_box[3], bounding_box[4], bounding_box[5] };

            for ( int dim = 0; dim < 3; ++dim ) {
                if ( !periodic[dim] && num_cell[dim] != 1 ) {
                    global_low_corner[dim] -= cell_size[dim] * halo_width;
                    global_high_corner[dim] += cell_size[dim] * halo_width;
                    num_cell[dim] += 2 * halo_width;
                    _min_domain_global_cell_index[dim] += halo_width;
                    _max_domain_global_cell_index[dim] -= halo_width;
                }
            }

            auto global_mesh = Cajita::createUniformGlobalMesh( global_low_corner, global_high_corner, cell_size );

            auto global_grid = Cajita::createGlobalGrid( comm, global_mesh, periodic, partitioner );

            /*
            printf("Global_Grid: %d, %d, %d\n", global_grid->ownedNumCell(0), global_grid->ownedNumCell(1), global_grid->ownedNumCell(2));
            */

            _local_grid = Cajita::createLocalGrid( global_grid, halo_width );
        }

        double cellSize() const {
            return _local_grid->globalGrid().globalMesh().cellSize( 0 );
        };

        std::array<int, 3> minDomainGlobalCellIndex() const {
            return _min_domain_global_cell_index;
        };

        std::array<int, 3> maxDomainGlobalCellIndex() const {
            return _max_domain_global_cell_index;
        };

        const std::shared_ptr<Cajita::LocalGrid<Cajita::UniformMesh<double>>>& localGrid() const {
            return _local_grid;
        };

    private:
        std::shared_ptr<Cajita::LocalGrid<Cajita::UniformMesh<double>>> _local_grid;

        std::array<int, 3> _min_domain_global_cell_index;
        std::array<int, 3> _max_domain_global_cell_index;
};

}

#endif

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
                MPI_Comm comm ) :
                _global_bounding_box ( global_bounding_box )
        {
            MPI_Comm_rank( comm, &_rank );

            // 2-D Mesh for now
            std::array<int, 3> num_cell;
            num_cell[0] = global_num_cell[0];
            num_cell[1] = global_num_cell[1];
            num_cell[2] = 1;

            for (int dim = 0; dim < 3; ++dim ) {
                _min_domain_global_node_index[dim] = 0;
                _max_domain_global_node_index[dim] = num_cell[dim] + 1;
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

            
            for ( int dim = 0; dim < 3; ++dim ) {
                double extent = num_cell[dim] * cell_size[dim];
                if ( std::abs( extent - ( bounding_box[dim + 3] - bounding_box[dim] ) ) > double( 10.0 ) * std::numeric_limits<double>::epsilon() )
                    throw std::logic_error( "Extent not evenly divisible by uniform cell size" );
            }
            
            
            if ( _rank == 0 ) {
                printf( "Num Cells: x: %d\ty: %d\tz: %d\n", num_cell[0], num_cell[1], num_cell[2] );
                printf( "Bounding Box: xl: %.4f\tyl: %.4f\tzl: %.4f\txu: %.4f\tyu: %.4f\tzu: %.4f\n", \
                bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3], bounding_box[4], bounding_box[5] );
                printf( "Cell Size: x: %.4f\ty: %.4f\tz: %.4f\n", cell_size[0], cell_size[1], cell_size[2] );
            }
            

            std::array<double, 3> global_low_corner = { bounding_box[0], bounding_box[1], bounding_box[2] };
            std::array<double, 3> global_high_corner = { bounding_box[3], bounding_box[4], bounding_box[5] };

            for ( int dim = 0; dim < 3; ++dim ) {
                if ( !periodic[dim] && num_cell[dim] != 1 ) {
                    global_low_corner[dim] -= cell_size[dim] * halo_width;
                    global_high_corner[dim] += cell_size[dim] * halo_width;
                    num_cell[dim] += 2 * halo_width;
                    _min_domain_global_node_index[dim] += halo_width;
                    _max_domain_global_node_index[dim] -= halo_width;
                }
            }

            auto global_mesh = Cajita::createUniformGlobalMesh( global_low_corner, global_high_corner, cell_size );

            auto global_grid = Cajita::createGlobalGrid( comm, global_mesh, periodic, partitioner );

            printf( "Global_Grid: Rank: %d\tNx: %d\tNy: %d\tNz: %d\tOffset X: %d\tOffset Y: %d\tOffset Z: %d\n", \
            global_grid->blockId(), global_grid->ownedNumCell( 0 ), global_grid->ownedNumCell( 1 ), global_grid->ownedNumCell( 2 ), \
            global_grid->globalOffset( 0 ), global_grid->globalOffset( 1 ), global_grid->globalOffset( 2 ) );

            _local_grid = Cajita::createLocalGrid( global_grid, halo_width );


        }

        double cellSize() const {
            return _local_grid->globalGrid().globalMesh().cellSize( 0 );
        };

        std::array<int, 3> minDomainGlobalNodeIndex() const {
            return _min_domain_global_node_index;
        };

        std::array<int, 3> maxDomainGlobalNodeIndex() const {
            return _max_domain_global_node_index;
        };

        const std::shared_ptr<Cajita::LocalGrid<Cajita::UniformMesh<double>>>& localGrid() const {
            return _local_grid;
        };

        const std::array<double, 6> globalBoundingBox() const {
            return _global_bounding_box;
        }

        int rank() const {
            return _rank;
        }

    private:
        int _rank;
        std::shared_ptr<Cajita::LocalGrid<Cajita::UniformMesh<double>>> _local_grid;

        const std::array<double, 6> _global_bounding_box;
        std::array<int, 3> _min_domain_global_node_index;
        std::array<int, 3> _max_domain_global_node_index;
};

}

#endif

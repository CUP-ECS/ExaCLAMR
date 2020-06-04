#ifndef EXACLAMR_MESH_HPP
#define EXACLAMR_MESH_HPP

#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include<mpi.h>

#include <memory>

namespace ExaCLAMR
{
template<class MemorySpace, class ExecutionSpace>
class Mesh
{
    public:

        Mesh( const std::array<double, 6>& global_bounding_box, 
                const std::array<int, 3>& global_num_cell,
                const std::array<bool, 3>& periodic,
                const Cajita::Partitioner& partitioner, 
                const int halo_width, 
                MPI_Comm comm,
                const ExecutionSpace& exec_space ) :
                _global_bounding_box ( global_bounding_box )
        {
            using device_type = typename Kokkos::Device<ExecutionSpace, MemorySpace>;

            MPI_Comm_rank( comm, &_rank );

            // 2-D Mesh for now
            std::array<int, 3> num_cell;
            num_cell[0] = global_num_cell[0];
            num_cell[1] = global_num_cell[1];
            num_cell[2] = 1;

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
                }
            }

            auto global_mesh = Cajita::createUniformGlobalMesh( global_low_corner, global_high_corner, cell_size );

            auto global_grid = Cajita::createGlobalGrid( comm, global_mesh, periodic, partitioner );

            printf( "Global_Grid: Rank: %d\tNx: %d\tNy: %d\tNz: %d\tOffset X: %d\tOffset Y: %d\tOffset Z: %d\n", \
            global_grid->blockId(), global_grid->ownedNumCell( 0 ), global_grid->ownedNumCell( 1 ), global_grid->ownedNumCell( 2 ), \
            global_grid->globalOffset( 0 ), global_grid->globalOffset( 1 ), global_grid->globalOffset( 2 ) );

            _local_grid = Cajita::createLocalGrid( global_grid, halo_width );

            auto local_mesh = Cajita::createLocalMesh<device_type>( *_local_grid );

            auto owned_cells = _local_grid->indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );

            _domainMin = { num_cell[0], num_cell[1], num_cell[2] };
            _domainMax = { 0, 0, 0 };
            

            Kokkos::parallel_for( Cajita::createExecutionPolicy( owned_cells, exec_space ), KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                int coords[3] = { i, j, k };
                double x[3];
                local_mesh.coordinates( Cajita::Cell(), coords, x );

                if ( x[0] >= global_bounding_box[0] && x[1] >= global_bounding_box[1] && x[2] >= global_bounding_box[2] 
                && x[0] <= global_bounding_box[3] && x[1] <= global_bounding_box[4] && x[2] <= global_bounding_box[5] ) {
                    _domainMin[0] = ( i < _domainMin[0] ) ? i : _domainMin[0];
                    _domainMin[1] = ( j < _domainMin[1] ) ? j : _domainMin[1];
                    _domainMin[2] = ( k < _domainMin[2] ) ? k : _domainMin[2];
                    _domainMax[0] = ( i + 1 > _domainMax[0] ) ? i + 1 : _domainMax[0];
                    _domainMax[1] = ( j + 1 > _domainMax[1] ) ? j + 1 : _domainMax[1];
                    _domainMax[2] = ( k + 1 > _domainMax[2] ) ? k + 1 : _domainMax[2];
                }
            } );
        };

        double cellSize() const {
            return _local_grid->globalGrid().globalMesh().cellSize( 0 );
        };

        const std::shared_ptr<Cajita::LocalGrid<Cajita::UniformMesh<double>>>& localGrid() const {
            return _local_grid;
        };

        const std::array<double, 6> globalBoundingBox() const {
            return _global_bounding_box;
        };

        int rank() const {
            return _rank;
        };

        const Cajita::IndexSpace<3> domainSpace() const {
            return Cajita::IndexSpace<3>( _domainMin, _domainMax );
        };

    private:
        int _rank;
        std::shared_ptr<Cajita::LocalGrid<Cajita::UniformMesh<double>>> _local_grid;
        std::array<long, 3> _domainMin;
        std::array<long, 3> _domainMax;
        const std::array<double, 6> _global_bounding_box;
};

}

#endif

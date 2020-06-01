#ifndef EXACLAMR_TIMEINTEGRATION_HPP
#define EXACLAMR_TIMEINTEGRATION_HPP

#include <ProblemManager.hpp>

#include <stdio.h>

namespace ExaCLAMR
{
namespace TimeIntegrator
{

template<class ProblemManagerType, class ExecutionSpace>
void applyBoundaryConditions( const ProblemManagerType& pm, const ExecutionSpace& exec_space ) {
    printf( "Applying Boundary Conditions\n" );
}

template<class ProblemManagerType, class ExecutionSpace, class MemorySpace, class state_t>
void step( const ProblemManagerType& pm, const ExecutionSpace& exec_space, const MemorySpace& mem_space, const state_t dt, const state_t gravity ) {
    printf( "Time Stepper\n" );

    typedef Kokkos::View< double **** > ViewType;

    using device_type = typename Kokkos::Device<ExecutionSpace, MemorySpace>;

    applyBoundaryConditions( pm, exec_space);

    auto local_grid = pm.mesh()->localGrid();
    auto local_mesh = Cajita::createLocalMesh<device_type>( *local_grid );

    auto global_grid = local_grid->globalGrid();
    auto global_mesh = global_grid.globalMesh();

    auto owned_cells = local_grid->indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );
    auto global_bounding_box = pm.mesh()->globalBoundingBox();

    auto u_i = pm.get( Location::Cell(), Field::Velocity() );
    auto h_i = pm.get( Location::Cell(), Field::Height() );

    ViewType height( "new height", owned_cells.max( 0 ), owned_cells.max( 1 ), owned_cells.max( 2 ), 1 );

    Kokkos::parallel_for( Cajita::createExecutionPolicy( owned_cells, exec_space ), KOKKOS_LAMBDA( const int i, const int j, const int k ) {
        int coords[3] = { i, j, k };
        state_t x[3];
        local_mesh.coordinates( Cajita::Cell(), coords, x );

        if ( x[0] >= global_bounding_box[0] && x[1] >= global_bounding_box[1] && x[2] >= global_bounding_box[2] 
        && x[0] <= global_bounding_box[3] && x[1] <= global_bounding_box[4] && x[2] <= global_bounding_box[5] ) {
            /*
            printf( "Rank: %d\tx: %.4f\ty: %.4f\tz: %.4f\n", pm.mesh()->rank(), x[0], x[1], x[2] );
            printf( "Rank: %d\ti: %d\tj: %d\tk: %d\n", pm.mesh()->rank(), i, j, k );
            */

            height( i, j, k, 0 ) = ( h_i( i - 1, j, k, 0 ) + h_i( i + 1, j, k, 0 ) + h_i( i, j - 1, k, 0 ) + h_i( i, j + 1, k, 0 ) ) / 4;
        }
    } );

    
    if ( pm.mesh()->rank() == 0 ) {
        for ( int i = 0; i < owned_cells.extent( 0 ); i++ ) {
            for ( int j = 0; j < owned_cells.extent( 1 ); j++ ) {
                for ( int k = 0; k < owned_cells.extent( 2 ); k++ ) {
                    printf( "%.4f\t", height( i, j, k, 0 ) );
                }
            }
            printf("\n");
        }
    }
    

}

}

}

#endif
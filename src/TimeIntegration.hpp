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

template<class ProblemManagerType>
void haloExchange( ProblemManagerType& pm ) {
    printf( "Starting Halo Exchange\n" );

    auto local_grid = pm.mesh()->localGrid();

    /*
    int requestCount = 0;

    for ( int i = -1; i < 2; i++ ) {
        for (int j = -1; j < 2; j++) {
            if ( ( i == 0 || j == 0 ) && !( i == 0 && j == 0 ) ){
                int neighbor = local_grid->neighborRank( i, j, 0 );
                if (neighbor != -1 ) {
                    requestCount += 6;
                }
            }
        }
    }

    printf( "Rank: %d\tRequest Count: %d\n", pm.mesh()->rank(), requestCount );

    MPI_Request request[requestCount];
    MPI_Status statuses[requestCount];
    */

    for ( int i = -1; i < 2; i++ ) {
        for (int j = -1; j < 2; j++) {
            if ( ( i == 0 || j == 0 ) && !( i == 0 && j == 0 ) ){
                int neighbor = local_grid->neighborRank( i, j, 0 );
                if (neighbor != -1 ) {
                    auto shared_recv_cells = local_grid->sharedIndexSpace( Cajita::Ghost(), Cajita::Cell(), i, j, 0 );
                    auto shared_send_cells = local_grid->sharedIndexSpace( Cajita::Own(), Cajita::Cell(), i, j, 0 );

                    printf( "Rank: %d\t i: %d\tj: %d\tk: %d\t Neighbor: %d\n", pm.mesh()->rank(), i, j, 0, neighbor );
                    printf( "Rank (Recv): %d\txmin: %d\t xmax: %d\tymin: %d\tymax: %d\tzmin: %d\tzmax: %d\tsize: %d\n", pm.mesh()->rank(), \
                    shared_recv_cells.min( 0 ), shared_recv_cells.max( 0 ), shared_recv_cells.min( 1 ), shared_recv_cells.max( 1 ), shared_recv_cells.min( 2 ), shared_recv_cells.max( 2 ), shared_recv_cells.size() );
                    printf( "Rank (Send): %d\txmin: %d\t xmax: %d\tymin: %d\tymax: %d\tzmin: %d\tzmax: %d\tsize: %d\n", pm.mesh()->rank(), \
                    shared_send_cells.min( 0 ), shared_send_cells.max( 0 ), shared_send_cells.min( 1 ), shared_send_cells.max( 1 ), shared_send_cells.min( 2 ), shared_send_cells.max( 2 ), shared_send_cells.size() );

                    /*
                    MPI_Request request[6];
                    MPI_Status statuses[6];

                    MPI_ISend( sendH, shared_cells_send.size(), MPI_DOUBLE, neighbor, 0, MPI_COMM_WORLD, &request[0] );
                    MPI_ISend( sendU, shared_cells_send.size(), MPI_DOUBLE, neighbor, 0, MPI_COMM_WORLD, &request[1] );
                    MPI_ISend( sendV, shared_cells_send.size(), MPI_DOUBLE, neighbor, 0, MPI_COMM_WORLD, &request[2] );

                    MPI_IRecv( recvH, shared_cells_recv.size(), MPI_DOUBLE, neighbor, 0, MPI_COMM_WORLD, &request[3] );
                    MPI_IRecv( recvU, shared_cells_recv.size(), MPI_DOUBLE, neighbor, 0, MPI_COMM_WORLD, &request[4] );
                    MPI_IRecv( recvV, shared_cells_recv.size(), MPI_DOUBLE, neighbor, 0, MPI_COMM_WORLD, &request[5] );

                    MPI_Waitall( 6, request, statuses );
                    */

                }
            }
        }
    }

    MPI_Barrier( MPI_COMM_WORLD );
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

    MPI_Barrier( MPI_COMM_WORLD );

    haloExchange( pm );

}

}

}

#endif
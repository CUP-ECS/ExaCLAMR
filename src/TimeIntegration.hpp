#ifndef EXACLAMR_TIMEINTEGRATION_HPP
#define EXACLAMR_TIMEINTEGRATION_HPP

#include <ProblemManager.hpp>

#include <stdio.h>

namespace ExaCLAMR
{
namespace TimeIntegrator
{

template<class ProblemManagerType, class ExecutionSpace, class MemorySpace, class state_t>
void applyBoundaryConditions( const ProblemManagerType& pm, const ExecutionSpace& exec_space, const MemorySpace& mem_space, const state_t gravity, int a, int b ) {
    if ( pm.mesh()->rank() == 0 ) printf( "Applying Boundary Conditions\n" );

    using device_type = typename Kokkos::Device<ExecutionSpace, MemorySpace>;

    auto uCurrent = pm.get( Location::Cell(), Field::Velocity(), a );
    auto hCurrent = pm.get( Location::Cell(), Field::Height(), a );

    auto uNew = pm.get( Location::Cell(), Field::Velocity(), b );
    auto hNew = pm.get( Location::Cell(), Field::Height(), b );

    auto local_grid = pm.mesh()->localGrid();
    auto local_mesh = Cajita::createLocalMesh<device_type>( *local_grid );

    auto owned_cells = local_grid->indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );
    auto global_bounding_box = pm.mesh()->globalBoundingBox();

    Kokkos::parallel_for( Cajita::createExecutionPolicy( owned_cells, exec_space ), KOKKOS_LAMBDA( const int i, const int j, const int k ) {
        int coords[3] = { i, j, k };
        state_t x[3];
        local_mesh.coordinates( Cajita::Cell(), coords, x );

        if ( x[0] < global_bounding_box[0] || x[1] < global_bounding_box[1] || x[2] < global_bounding_box[2] 
        || x[0] > global_bounding_box[3] || x[1] > global_bounding_box[4] || x[2] > global_bounding_box[5] ) {
            // printf("i: %d\tj: %d\tk: %d\n", i, j, k);

        }
    } );
}

template<class ProblemManagerType>
void haloExchange( const ProblemManagerType& pm, const int a, const int b ) {
    if ( pm.mesh()->rank() == 0 ) printf( "Starting Halo Exchange\ta: %d\t b: %d\n", a, b );

    pm.gather( Location::Cell(), Field::Velocity(), b );
    pm.gather( Location::Cell(), Field::Height(), b );

    MPI_Barrier( MPI_COMM_WORLD );
}

double setTimeStep( const double gravity, const double sigma ) {
    return 0.0;
}

#define HXRGFLUXIC ( uCurrent( i, j, k, 0 ) )
#define HXRGFLUXNL ( uCurrent( i - 1, j, k, 0 ) )
#define HXRGFLUXNR ( uCurrent( i + 1, j, k, 0 ) )
#define HXRGFLUXNT ( uCurrent( i, j - 1, k, 0 ) )
#define HXRGFLUXNB ( uCurrent( i, j + 1, k, 0 ) )

#define UXRGFLUXIC ( sqrt( uCurrent( i, j, k, 0 ) )     / hCurrent( i, j, k, 0 )     + ghalf * sqrt( hCurrent( i, j, k, 0 ) ) )
#define UXRGFLUXNL ( sqrt( uCurrent( i - 1, j, k, 0 ) ) / hCurrent( i - 1, j, k, 0 ) + ghalf * sqrt( hCurrent( i - 1, j, k, 0 ) ) )
#define UXRGFLUXNR ( sqrt( uCurrent( i + 1, j, k, 0 ) ) / hCurrent( i + 1, j, k, 0 ) + ghalf * sqrt( hCurrent( i + 1, j, k, 0 ) ) )
#define UXRGFLUXNB ( sqrt( uCurrent( i, j - 1, k, 0 ) ) / hCurrent( i, j - 1, k, 0 ) + ghalf * sqrt( hCurrent( i, j - 1, k, 0 ) ) )
#define UXRGFLUXNT ( sqrt( uCurrent( i, j + 1, k, 0 ) ) / hCurrent( i, j + 1, k, 0 ) + ghalf * sqrt( hCurrent( i, j + 1, k, 0 ) ) )

#define VXRGFLUXIC ( uCurrent( i, j, k, 0 )     * uCurrent( i, j, k, 1 )     / hCurrent( i, j, k, 0 ) )
#define VXRGFLUXNL ( uCurrent( i - 1, j, k, 0 ) * uCurrent( i - 1, j, k, 1 ) / hCurrent( i - 1, j, k, 0 ) )
#define VXRGFLUXNR ( uCurrent( i + 1, j, k, 0 ) * uCurrent( i + 1, j, k, 1 ) / hCurrent( i + 1, j, k, 0 ) )
#define VXRGFLUXNB ( uCurrent( i, j - 1, k, 0 ) * uCurrent( i, j - 1, k, 1 ) / hCurrent( i, j - 1, k, 0 ) )
#define VXRGFLUXNT ( uCurrent( i, j + 1, k, 0 ) * uCurrent( i, j + 1, k, 1 ) / hCurrent( i, j + 1, k, 0 ) )

#define HYRGFLUXIC ( uCurrent( i, j, k, 1 ) )
#define HYRGFLUXNL ( uCurrent( i - 1, j, k, 1 ) )
#define HYRGFLUXNR ( uCurrent( i + 1, j, k, 1 ) )
#define HYRGFLUXNB ( uCurrent( i, j - 1, k, 1 ) )
#define HYRGFLUXNT ( uCurrent( i, j + 1, k, 1 ) )

#define UYRGFLUXIC ( uCurrent( i, j, k, 1 )     * uCurrent( i, j, k, 0 )     / hCurrent( i, j, k, 0 ) )
#define UYRGFLUXNL ( uCurrent( i - 1, j, k, 1 ) * uCurrent( i - 1, j, k, 0 ) / hCurrent( i - 1, j, k, 0 ) )
#define UYRGFLUXNR ( uCurrent( i + 1, j, k, 1 ) * uCurrent( i + 1, j, k, 0 ) / hCurrent( i + 1, j, k, 0 ) )
#define UYRGFLUXNB ( uCurrent( i, j - 1, k, 1 ) * uCurrent( i, j - 1, k, 0 ) / hCurrent( i, j - 1, k, 0 ) )
#define UYRGFLUXNT ( uCurrent( i, j + 1, k, 1 ) * uCurrent( i, j + 1, k, 0 ) / hCurrent( i, j + 1, k, 0 ) )

#define VYRGFLUXIC ( sqrt( uCurrent( i, j, k, 1 ) )     / hCurrent( i, j, k, 0 )     + ghalf * sqrt( hCurrent( i, j, k, 0 ) ) )
#define VYRGFLUXNL ( sqrt( uCurrent( i - 1, j, k, 1 ) ) / hCurrent( i - 1, j, k, 0 ) + ghalf * sqrt( hCurrent( i - 1, j, k, 0 ) ) )
#define VYRGFLUXNR ( sqrt( uCurrent( i + 1, j, k, 1 ) ) / hCurrent( i + 1, j, k, 0 ) + ghalf * sqrt( hCurrent( i + 1, j, k, 0 ) ) )
#define VYRGFLUXNB ( sqrt( uCurrent( i, j - 1, k, 1 ) ) / hCurrent( i, j - 1, k, 0 ) + ghalf * sqrt( hCurrent( i, j - 1, k, 0 ) ) )
#define VYRGFLUXNT ( sqrt( uCurrent( i, j + 1, k, 1 ) ) / hCurrent( i, j + 1, k, 0 ) + ghalf * sqrt( hCurrent( i, j + 1, k, 0 ) ) )

template<class state_t>
inline state_t wCorrector( state_t dt, state_t dr, state_t uEigen, state_t gradHalf, state_t gradMinus, state_t gradPlus ) {
    state_t nu = 0.5 * uEigen * dt / dr;
    nu *= ( 1.0 - nu );

    state_t rDenom = 1.0 / std::max( std::pow( gradHalf, 2 ), 1.0e-30 );
    state_t rPlus = ( gradPlus * gradHalf ) * rDenom;
    state_t rMinus = ( gradMinus * gradHalf ) * rDenom;

    return 0.5 * nu * ( 1.0 - std::max( std::min( std::min( 1.0, rPlus ), rMinus ), 0.0 ) );
}

template<class state_t>
inline state_t uFullStep( state_t dt, state_t dr, state_t U, state_t FPlus, state_t FMinus, state_t GPlus, state_t GMinus ) {
    return ( U + ( - ( dt / dr) * ( ( FPlus - FMinus ) + ( GPlus - GMinus ) ) ) );
}

template<class ProblemManagerType, class ExecutionSpace, class MemorySpace, class state_t>
void step( const ProblemManagerType& pm, const ExecutionSpace& exec_space, const MemorySpace& mem_space, const state_t dt, const state_t gravity, const int tstep ) {
    if ( pm.mesh()->rank() == 0 ) printf( "Time Stepper\n" );

    using device_type = typename Kokkos::Device<ExecutionSpace, MemorySpace>;

    double dx = pm.mesh()->localGrid()->globalGrid().globalMesh().cellSize( 0 );
    double dy = pm.mesh()->localGrid()->globalGrid().globalMesh().cellSize( 1 );
    double ghalf = 0.5 * gravity;

    auto local_grid = pm.mesh()->localGrid();
    auto local_mesh = Cajita::createLocalMesh<device_type>( *local_grid );

    auto owned_cells = local_grid->indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );
    auto global_bounding_box = pm.mesh()->globalBoundingBox();

    int a, b;
    if ( tstep % 2 == 0 ) {
        a = 0;
        b = 1;
    }
    else {
        a = 1;
        b = 0;
    }

    applyBoundaryConditions( pm, exec_space, mem_space, gravity, a, b );

    auto uCurrent = pm.get( Location::Cell(), Field::Velocity(), a );
    auto hCurrent = pm.get( Location::Cell(), Field::Height(), a );

    auto uNew = pm.get( Location::Cell(), Field::Velocity(), b );
    auto hNew = pm.get( Location::Cell(), Field::Height(), b );

    auto HxFluxPlus = pm.get( Location::Cell(), Field::HxFluxPlus() );
    auto HxFluxMinus = pm.get( Location::Cell(), Field::HxFluxMinus() );
    auto UxFluxPlus = pm.get( Location::Cell(), Field::UxFluxPlus() );
    auto UxFluxMinus = pm.get( Location::Cell(), Field::UxFluxMinus() );

    auto HyFluxPlus = pm.get( Location::Cell(), Field::HyFluxPlus() );
    auto HyFluxMinus = pm.get( Location::Cell(), Field::HyFluxMinus() );
    auto UyFluxPlus = pm.get( Location::Cell(), Field::UyFluxPlus() );
    auto UyFluxMinus = pm.get( Location::Cell(), Field::UyFluxMinus() );

    auto HxWPlus = pm.get( Location::Cell(), Field::HxWPlus() );
    auto HxWMinus = pm.get( Location::Cell(), Field::HxWMinus() );
    auto HyWPlus = pm.get( Location::Cell(), Field::HyWPlus() );
    auto HyWMinus = pm.get( Location::Cell(), Field::HyWMinus() );

    auto UWPlus = pm.get( Location::Cell(), Field::UWPlus() );
    auto UWMinus = pm.get( Location::Cell(), Field::UWMinus() );

    auto domain = pm.mesh()->domainSpace();
    printf("Domain: %ld\t%ld\t%ld\t%ld\t%ld\t%ld\n", domain.min( 0 ), domain.min( 1 ), domain.min( 2 ), domain.max( 0 ), domain.max( 1 ), domain.max( 2 ));

    Kokkos::parallel_for( Cajita::createExecutionPolicy( domain, exec_space ), KOKKOS_LAMBDA( const int i, const int j, const int k ) {
        int coords[3] = { i, j, k };
        state_t x[3];
        local_mesh.coordinates( Cajita::Cell(), coords, x );
            
        /*
        printf( "Rank: %d\tx: %.4f\ty: %.4f\tz: %.4f\n", pm.mesh()->rank(), x[0], x[1], x[2] );
        printf( "Rank: %d\ti: %d\tj: %d\tk: %d\n", pm.mesh()->rank(), i, j, k );
        */
            
        // Simple Diffusion Problem
        hNew( i, j, k, 0 ) = ( hCurrent( i - 1, j, k, 0 ) + hCurrent( i + 1, j, k, 0 ) + hCurrent( i, j - 1, k, 0 ) + hCurrent( i, j + 1, k, 0 ) ) / 4;

        /*
        // Shallow Water Equations
        state_t HxMinus = 0.5 * ( ( hCurrent( i - 1, j, k, 0 ) + hCurrent( i, j, k, 0 ) ) - ( dt ) / ( dx ) * ( ( HXRGFLUXIC ) - ( HXRGFLUXNL ) ) );
        state_t UxMinus = 0.5 * ( ( uCurrent( i - 1, j, k, 0 ) + uCurrent( i, j, k, 0 ) ) - ( dt ) / ( dx ) * ( ( UXRGFLUXIC ) - ( UXRGFLUXNL ) ) );
        state_t VxMinus = 0.5 * ( ( uCurrent( i - 1, j, k, 1 ) + uCurrent( i, j, k, 1 ) ) - ( dt ) / ( dx ) * ( ( VXRGFLUXIC ) - ( VXRGFLUXNL ) ) );

        // printf( "%-10s: %-.4f\t%-10s: %.4f\t%-10s: %.3f\ti: %d\tj:%d\tk: %d\n",  "HxMinus", HxMinus, "UxMinus", UxMinus, "VxMinus", VxMinus, i, j, k );

        state_t HxPlus  = 0.5 * ( ( hCurrent( i, j, k, 0 ) + hCurrent( i + 1, j, k, 0 ) ) - ( dt ) / ( dx ) * ( ( HXRGFLUXNR ) - ( HXRGFLUXIC ) ) );
        state_t UxPlus  = 0.5 * ( ( uCurrent( i, j, k, 0 ) + uCurrent( i + 1, j, k, 0 ) ) - ( dt ) / ( dx ) * ( ( UXRGFLUXNR ) - ( UXRGFLUXIC ) ) );
        state_t VxPlus  = 0.5 * ( ( uCurrent( i, j, k, 1 ) + uCurrent( i + 1, j, k, 1 ) ) - ( dt ) / ( dx ) * ( ( VXRGFLUXNR ) - ( VXRGFLUXIC ) ) );

        // printf( "%-10s: %-.4f\t%-10s: %.4f\t%-10s: %.3f\ti: %d\tj:%d\tk: %d\n",  "HxPlus", HxPlus, "UxPlus", UxPlus, "VxPlus", VxPlus, i, j, k );

        state_t HyMinus = 0.5 * ( ( hCurrent( i, j - 1, k, 0 ) + hCurrent( i, j, k, 0 ) ) - ( dt ) / ( dy ) * ( ( HYRGFLUXIC ) - ( HYRGFLUXNB ) ) );
        state_t UyMinus = 0.5 * ( ( uCurrent( i, j - 1, k, 0 ) + uCurrent( i, j, k, 0 ) ) - ( dt ) / ( dy ) * ( ( UYRGFLUXIC ) - ( UYRGFLUXNB ) ) );
        state_t VyMinus = 0.5 * ( ( uCurrent( i, j - 1, k, 1 ) + uCurrent( i, j, k, 1 ) ) - ( dt ) / ( dy ) * ( ( VYRGFLUXIC ) - ( VYRGFLUXNB ) ) );

        // printf( "%-10s: %-.4f\t%-10s: %.4f\t%-10s: %.3f\ti: %d\tj:%d\tk: %d\n",  "HxPlus", HxPlus, "UxPlus", UxPlus, "VxPlus", VxPlus, i, j, k );

        state_t HyPlus  = 0.5 * ( ( hCurrent( i, j, k, 0 ) + hCurrent( i, j + 1, k, 0 ) ) - ( dt ) / ( dy ) * ( ( HYRGFLUXNT ) - ( HYRGFLUXIC ) ) );
        state_t UyPlus  = 0.5 * ( ( uCurrent( i, j, k, 0 ) + uCurrent( i, j + 1, k, 0 ) ) - ( dt ) / ( dy ) * ( ( UYRGFLUXNT ) - ( UYRGFLUXIC ) ) );
        state_t VyPlus  = 0.5 * ( ( uCurrent( i, j, k, 1 ) + uCurrent( i, j + 1, k, 1 ) ) - ( dt ) / ( dy ) * ( ( VYRGFLUXNT ) - ( VYRGFLUXIC ) ) );

        // printf( "%-10s: %-.4f\t%-10s: %.4f\t%-10s: %.3f\ti: %d\tj:%d\tk: %d\n",  "HxPlus", HxPlus, "UxPlus", UxPlus, "VxPlus", VxPlus, i, j, k );

        HxFluxMinus( i, j, k, 0 ) = UxMinus;
        UxFluxMinus( i, j, k, 0 ) = ( sqrt( UxMinus ) / HxMinus + ghalf * sqrt( HxMinus ) );
        UxFluxMinus( i, j, k, 1 ) = UxMinus * VxMinus / HxMinus;

        HxFluxPlus( i, j, k, 0 ) = UxPlus;
        UxFluxPlus( i, j, k, 0 ) = ( sqrt( UxPlus ) / HxPlus + ghalf * sqrt( HxPlus ) );
        UxFluxPlus( i ,j, k, 1 ) = ( UxPlus * VxPlus / HxPlus );

        HyFluxMinus( i, j, k, 0 ) = VyMinus;
        UyFluxMinus( i, j, k, 0 ) = ( VyMinus * UyMinus / HyMinus );
        UyFluxMinus( i, j, k, 1 ) = ( sqrt( VyMinus ) / HyMinus + ghalf * sqrt( HyMinus ) );

        HyFluxPlus( i, j, k, 0 ) = VyPlus;
        UyFluxPlus( i, j, k, 0 ) = ( VyPlus * UyPlus / HyPlus );
        UyFluxPlus( i, j, k, 1 ) = ( sqrt( VyPlus ) / HyPlus + ghalf * sqrt( HyPlus ) );


        HxWMinus( i, j, k, 0 ) = wCorrector( dt, dx, fabs( UxMinus / HxMinus ) + sqrt( gravity * HxMinus ), hCurrent( i, j, k, 0 ) - hCurrent( i - 1, j, k, 0 ), hCurrent( i - 1, j, k, 0 ) - hCurrent( i - 2, j, k, 0 ), hCurrent( i + 1, j, k, 0 ) - hCurrent( i, j, k, 0 ) );
        HxWMinus( i, j, k, 0 ) *= hCurrent( i, j, k, 0 ) - hCurrent( i - 1, j, k, 0 );
        
        HxWPlus( i, j, k, 0 )  = wCorrector( dt, dx, fabs( UxPlus / HxPlus ) + sqrt( gravity * HxPlus ), hCurrent( i + 1, j, k, 0 ) - hCurrent( i, j, k, 0 ), hCurrent( i, j, k, 0 ) - hCurrent( i - 1, j, k, 0 ), hCurrent( i + 2, j, k, 0 ) - hCurrent( i + 1, j, k, 0 ) );
        HxWPlus( i, j, k, 0 )  *= hCurrent( i + 1, j, k, 0 ) - hCurrent( i, j, k, 0 );

        UWMinus( i, j, k, 0 )  = wCorrector( dt, dx, fabs( UxMinus / HxMinus ) + sqrt( gravity * HxMinus ), uCurrent( i, j, k, 0 ) - uCurrent( i - 1, j, k, 0 ), uCurrent( i - 1, j, k, 0 ) - uCurrent( i - 2, j, k, 0 ), uCurrent( i + 1, j, k, 0 ) - uCurrent( i, j, k, 0 ) );
        UWMinus( i, j, k, 0 )  *= uCurrent( i, j, k, 0 ) - uCurrent( i - 1, j, k, 0 );

        UWPlus( i, j, k, 0 )   = wCorrector( dt, dx, fabs( UxPlus / HxPlus ) + sqrt( gravity * HxPlus ), uCurrent( i + 1, j, k, 0 ) - uCurrent( i, j, k, 0 ), uCurrent( i, j, k, 0 ) - uCurrent( i - 1, j, k, 0 ), uCurrent( i + 2, j, k, 0 ) - uCurrent( i + 1, j, k, 0 ) );
        UWPlus( i, j, k, 0 )   *= uCurrent( i + 1, j, k, 0 ) - uCurrent( i, j, k, 0 );


        HyWMinus( i, j, k, 0 ) = wCorrector( dt, dy, fabs( VyMinus / HyMinus ) + sqrt( gravity * HyMinus ), hCurrent( i, j, k, 0) - hCurrent( i, j - 1, k, 0 ), hCurrent( i, j - 1, k, 0 ) - hCurrent( i, j - 2, k, 0 ), hCurrent( i, j + 1, k, 0 ) - hCurrent( i, j, k, 0) );
        HyWMinus( i, j, k, 0 ) *= hCurrent( i, j, k, 0 ) - hCurrent( i, j - 1, k, 0 );

        HyWPlus( i, j, k, 0 )  = wCorrector( dt, dy, fabs( VyPlus / HyPlus ) + sqrt( gravity * HyPlus ), hCurrent( i, j + 1, k, 0) - hCurrent( i, j, k, 0 ), hCurrent( i, j, k, 0 ) - hCurrent( i, j - 1, k, 0 ), hCurrent( i, j + 2, k, 0 ) - hCurrent( i, j + 1, k, 0)  );
        HyWPlus( i, j, k, 0 )  *= hCurrent( i, j + 1, k, 0 ) - hCurrent( i, j, k, 0 );

        UWMinus( i, j, k, 1 )  = wCorrector( dt, dy, fabs( VyMinus / HyMinus ) + sqrt( gravity * HyMinus ), uCurrent( i, j, k, 1 ) - uCurrent( i, j - 1, k, 1 ), uCurrent( i, j - 1, k, 1) - uCurrent( i, j - 2, k, 1 ), uCurrent( i, j + 1, k, 1 ) - uCurrent( i, j, k, 1 ) );
        UWMinus( i, j, k, 1 )  *= uCurrent( i, j, k, 1 ) - uCurrent( i, j - 1, k, 1 );

        UWPlus( i, j, k, 1 )   = wCorrector( dt, dy, fabs( VyPlus / HyPlus ) + sqrt( gravity * HyPlus ), uCurrent( i, j + 1, k, 1 ) - uCurrent( i, j, k, 1 ), uCurrent( i, j, k, 1) - uCurrent( i, j - 1, k, 1 ), uCurrent( i, j, k, 1 ) - uCurrent( i, j + 1, k, 1 ) );
        UWPlus( i, j, k, 1 )   *= uCurrent( i, j + 1, k, 1 ) - uCurrent( i, j, k, 1 );


        hNew ( i, j, k, 0 ) = uFullStep( dt, dx, hCurrent( i, j, k, 0 ), HxFluxPlus( i, j, k, 0 ), HxFluxMinus( i, j, k, 0 ), HyFluxPlus( i, j, k, 0 ), HyFluxMinus( i, j, k, 0 ) ) - HxWMinus( i, j, k, 0 ) + HxWPlus( i, j, k, 0 ) - HyWMinus( i, j, k, 0 ) + HyWPlus( i, j, k, 0 );
        uNew ( i, j, k, 0 ) = uFullStep( dt, dx, uCurrent( i, j, k, 0 ), UxFluxPlus( i, j, k, 0 ), UxFluxMinus( i, j, k, 0 ), UyFluxPlus( i, j, k, 0 ), UyFluxMinus( i, j, k, 0 ) ) - UWMinus( i, j, k, 0 ) + UWPlus( i, j, k, 0 );
        uNew ( i, j, k, 1 ) = uFullStep( dt, dx, uCurrent( i, j, k, 1 ), UxFluxPlus( i, j, k, 1 ), UxFluxMinus( i, j, k, 1 ), UyFluxPlus( i, j, k, 1 ), UyFluxMinus( i, j, k, 1 ) ) - UWMinus( i, j, k, 1 ) + UWPlus( i, j, k, 1 );
`           */
    } );

    MPI_Barrier( MPI_COMM_WORLD );

    haloExchange( pm, a, b );

}

}

}

#endif
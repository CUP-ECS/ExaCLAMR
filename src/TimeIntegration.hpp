#ifndef EXACLAMR_TIMEINTEGRATION_HPP
#define EXACLAMR_TIMEINTEGRATION_HPP

#define DEBUG 0 

#include <ProblemManager.hpp>

#include <stdio.h>

namespace ExaCLAMR
{
namespace TimeIntegrator
{

template<class ProblemManagerType, class ExecutionSpace, class MemorySpace, class state_t>
void applyBoundaryConditions( const ProblemManagerType& pm, const ExecutionSpace& exec_space, const MemorySpace& mem_space, const state_t gravity, int a, int b ) {
    if ( pm.mesh()->rank() == 0 && DEBUG ) printf( "Applying Boundary Conditions\n" );

    using device_type = typename Kokkos::Device<ExecutionSpace, MemorySpace>;

    auto uCurrent = pm.get( Location::Cell(), Field::Velocity(), a );
    auto hCurrent = pm.get( Location::Cell(), Field::Height(), a );

    auto uNew = pm.get( Location::Cell(), Field::Velocity(), b );
    auto hNew = pm.get( Location::Cell(), Field::Height(), b );

    auto local_grid = pm.mesh()->localGrid();
    auto local_mesh = Cajita::createLocalMesh<device_type>( *local_grid );

    auto owned_cells = local_grid->indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );
    auto global_bounding_box = pm.mesh()->globalBoundingBox();

    auto domain = pm.mesh()->domainSpace();

    Kokkos::parallel_for( Cajita::createExecutionPolicy( owned_cells, exec_space ), KOKKOS_LAMBDA( const int i, const int j, const int k ) {
        int coords[3] = { i, j, k };
        state_t x[3];
        local_mesh.coordinates( Cajita::Cell(), coords, x );

        if ( j == domain.min( 1 ) - 1 && i >= domain.min( 0 ) && i <= domain.max( 0 ) - 1 ) {
            if ( DEBUG ) printf("Rank: %d\tLeft Boundary:\ti: %d\tj: %d\tk: %d\n", pm.mesh()->rank(), i, j, k);
            hCurrent( i, j, k, 0 ) = hCurrent( i, j + 1, k, 0 );
            uCurrent( i, j, k, 0 ) = uCurrent( i, j + 1, k, 0 );
            uCurrent( i, j, k, 1 ) = -uCurrent( i, j + 1, k, 1 );

            hCurrent( i, j - 1, k, 0 ) = 0;
            uCurrent( i, j - 1, k, 0 ) = 0;
            uCurrent( i, j - 1, k, 1 ) = 0;
        }
        
        if ( j == domain.max( 1 ) && i >= domain.min( 0 ) && i <= domain.max( 0 ) - 1 ) {
            if ( DEBUG )printf("Rank: %d\tRight Boundary:\ti: %d\tj: %d\tk: %d\n", pm.mesh()->rank(), i, j, k);
            hCurrent( i, j, k, 0 ) = hCurrent( i, j - 1, k, 0 );
            uCurrent( i, j, k, 0 ) = uCurrent( i, j - 1, k, 0 );
            uCurrent( i, j, k, 1 ) = -uCurrent( i, j - 1, k, 1 );

            hCurrent( i, j + 1, k, 0 ) = 0;
            uCurrent( i, j + 1, k, 0 ) = 0;
            uCurrent( i, j + 1, k, 1 ) = 0;
        }

        if ( i == domain.max( 0 ) && j >= domain.min( 1 ) && j <= domain.max( 1 ) - 1 ) {
            if ( DEBUG ) printf("Rank: %d\tBottom Boundary:\ti: %d\tj: %d\tk: %d\n", pm.mesh()->rank(), i, j, k);
            hCurrent( i, j, k, 0 ) = hCurrent( i - 1, j, k, 0 );
            uCurrent( i, j, k, 0 ) = -uCurrent( i - 1, j, k, 0 );
            uCurrent( i, j, k, 1 ) = uCurrent( i - 1, j, k, 1 );

            hCurrent( i + 1, j, k, 0 ) = 0;
            uCurrent( i + 1, j, k, 0 ) = 0;
            uCurrent( i + 1, j, k, 1 ) = 0;
        }

        if ( i == domain.min( 0 ) - 1 && j >= domain.min( 1 ) && j <= domain.max( 1 ) - 1 ) {
            if ( DEBUG ) printf("Rank: %d\tTop Boundary:\ti: %d\tj: %d\tk: %d\n", pm.mesh()->rank(), i, j, k);
            hCurrent( i, j, k, 0 ) = hCurrent( i + 1, j, k, 0 );
            uCurrent( i, j, k, 0 ) = -uCurrent( i + 1, j, k, 0 );
            uCurrent( i, j, k, 1 ) = uCurrent( i + 1, j, k, 1 );

            hCurrent( i - 1, j, k, 0 ) = 0;
            uCurrent( i - 1, j, k, 0 ) = 0;
            uCurrent( i - 1, j, k, 1 ) = 0;
        }
    } );

    MPI_Barrier( MPI_COMM_WORLD );

}

template<class ProblemManagerType>
void haloExchange( const ProblemManagerType& pm, const int a, const int b ) {
    if ( pm.mesh()->rank() == 0 && DEBUG ) printf( "Starting Halo Exchange\ta: %d\t b: %d\n", a, b );

    pm.gather( Location::Cell(), Field::Velocity(), b );
    pm.gather( Location::Cell(), Field::Height(), b );

    MPI_Barrier( MPI_COMM_WORLD );
}

template<class ProblemManagerType, class ExecutionSpace, class MemorySpace, class state_t>
state_t setTimeStep( const ProblemManagerType& pm, const ExecutionSpace& exec_space, const MemorySpace& mem_space, const state_t gravity, const state_t sigma, const int a, const int b ) {

    double dx = pm.mesh()->localGrid()->globalGrid().globalMesh().cellSize( 0 );
    double dy = pm.mesh()->localGrid()->globalGrid().globalMesh().cellSize( 1 );

    auto uCurrent = pm.get( Location::Cell(), Field::Velocity(), a );
    auto hCurrent = pm.get( Location::Cell(), Field::Height(), a );

    auto uNew = pm.get( Location::Cell(), Field::Velocity(), b );
    auto hNew = pm.get( Location::Cell(), Field::Height(), b );

    auto domain = pm.mesh()->domainSpace();

    state_t mymindeltaT;

    Kokkos::parallel_reduce( Cajita::createExecutionPolicy( domain, exec_space ), KOKKOS_LAMBDA( const int i, const int j, const int k, state_t& lmin ) {
        state_t wavespeed = sqrt( gravity * hCurrent( i, j, k, 0 ) );
        state_t xspeed = ( fabs( uCurrent( i, j, k, 0 ) + wavespeed ) ) / dx;
        state_t yspeed = ( fabs( uCurrent( i, j, k, 1 ) + wavespeed ) ) / dy;

        state_t deltaT = sigma / ( xspeed + yspeed );

        if ( DEBUG ) printf( "Wavespeed: %.4f\txspeed: %.4f\tyspeed: %.4f\tdt: %.4f\n", wavespeed, xspeed, yspeed, deltaT );

        if ( deltaT < lmin ) lmin = deltaT;
    }, Kokkos::Min<state_t>( mymindeltaT ) );

    if ( DEBUG ) printf( "dt: %.4f\n", mymindeltaT );

    return mymindeltaT;
}

#define POW2( x ) ( ( x ) * ( x ) )

#define HXRGFLUXIC ( uCurrent( i, j, k, 0 ) )
#define HXRGFLUXNL ( uCurrent( i - 1, j, k, 0 ) )
#define HXRGFLUXNR ( uCurrent( i + 1, j, k, 0 ) )
#define HXRGFLUXNT ( uCurrent( i, j - 1, k, 0 ) )
#define HXRGFLUXNB ( uCurrent( i, j + 1, k, 0 ) )

#define UXRGFLUXIC ( POW2( uCurrent( i, j, k, 0 ) )     / hCurrent( i, j, k, 0 )     + ghalf * POW2( hCurrent( i, j, k, 0 ) ) )
#define UXRGFLUXNL ( POW2( uCurrent( i - 1, j, k, 0 ) ) / hCurrent( i - 1, j, k, 0 ) + ghalf * POW2( hCurrent( i - 1, j, k, 0 ) ) )
#define UXRGFLUXNR ( POW2( uCurrent( i + 1, j, k, 0 ) ) / hCurrent( i + 1, j, k, 0 ) + ghalf * POW2( hCurrent( i + 1, j, k, 0 ) ) )
#define UXRGFLUXNB ( POW2( uCurrent( i, j - 1, k, 0 ) ) / hCurrent( i, j - 1, k, 0 ) + ghalf * POW2( hCurrent( i, j - 1, k, 0 ) ) )
#define UXRGFLUXNT ( POW2( uCurrent( i, j + 1, k, 0 ) ) / hCurrent( i, j + 1, k, 0 ) + ghalf * POW2( hCurrent( i, j + 1, k, 0 ) ) )

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

#define VYRGFLUXIC ( POW2( uCurrent( i, j, k, 1 ) )     / hCurrent( i, j, k, 0 )     + ghalf * POW2( hCurrent( i, j, k, 0 ) ) )
#define VYRGFLUXNL ( POW2( uCurrent( i - 1, j, k, 1 ) ) / hCurrent( i - 1, j, k, 0 ) + ghalf * POW2( hCurrent( i - 1, j, k, 0 ) ) )
#define VYRGFLUXNR ( POW2( uCurrent( i + 1, j, k, 1 ) ) / hCurrent( i + 1, j, k, 0 ) + ghalf * POW2( hCurrent( i + 1, j, k, 0 ) ) )
#define VYRGFLUXNB ( POW2( uCurrent( i, j - 1, k, 1 ) ) / hCurrent( i, j - 1, k, 0 ) + ghalf * POW2( hCurrent( i, j - 1, k, 0 ) ) )
#define VYRGFLUXNT ( POW2( uCurrent( i, j + 1, k, 1 ) ) / hCurrent( i, j + 1, k, 0 ) + ghalf * POW2( hCurrent( i, j + 1, k, 0 ) ) )

template<class state_t>
inline state_t wCorrector( state_t dt, state_t dr, state_t uEigen, state_t gradHalf, state_t gradMinus, state_t gradPlus ) {
    state_t nu = 0.5 * uEigen * dt / dr;
    nu *= ( 1.0 - nu );

    state_t rDenom = 1.0 / std::max( POW2( gradHalf ), 1.0e-30 );
    state_t rPlus = ( gradPlus * gradHalf ) * rDenom;
    state_t rMinus = ( gradMinus * gradHalf ) * rDenom;

    return 0.5 * nu * ( 1.0 - std::max( std::min( std::min( 1.0, rPlus ), rMinus ), 0.0 ) );
}

template<class state_t>
inline state_t uFullStep( state_t dt, state_t dr, state_t U, state_t FPlus, state_t FMinus, state_t GPlus, state_t GMinus ) {
    return ( U + ( - ( dt / dr) * ( ( FPlus - FMinus ) + ( GPlus - GMinus ) ) ) );
}

template<class ProblemManagerType, class ExecutionSpace, class MemorySpace, class state_t>
void step( const ProblemManagerType& pm, const ExecutionSpace& exec_space, const MemorySpace& mem_space, const state_t dt, const state_t gravity, const int a, const int b ) {
    if ( pm.mesh()->rank() == 0 && DEBUG ) printf( "Time Stepper\n" );

    using device_type = typename Kokkos::Device<ExecutionSpace, MemorySpace>;

    double dx = pm.mesh()->localGrid()->globalGrid().globalMesh().cellSize( 0 );
    double dy = pm.mesh()->localGrid()->globalGrid().globalMesh().cellSize( 1 );
    double ghalf = 0.5 * gravity;

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

    auto domainSpace = pm.mesh()->domainSpace();

    if ( DEBUG ) printf("DomainSpace: %ld\t%ld\t%ld\t%ld\t%ld\t%ld\n", domainSpace.min( 0 ), domainSpace.min( 1 ), domainSpace.min( 2 ), domainSpace.max( 0 ), domainSpace.max( 1 ), domainSpace.max( 2 ));

    Kokkos::parallel_for( Cajita::createExecutionPolicy( domainSpace, exec_space ), KOKKOS_LAMBDA( const int i, const int j, const int k ) {            
        if ( DEBUG ) {
            auto local_mesh = Cajita::createLocalMesh<device_type>( * pm.mesh()->localGrid() );
            int coords[3] = { i, j, k };
            double x[3];
            local_mesh.coordinates( Cajita::Cell(), coords, x );
            printf( "Rank: %d\ti: %d\tj: %d\tk: %d\n", pm.mesh()->rank(), i, j, k, x[0], x[1], x[2] );
        }
            
        // Simple Diffusion Problem
        // hNew( i, j, k, 0 ) = ( hCurrent( i - 1, j, k, 0 ) + hCurrent( i + 1, j, k, 0 ) + hCurrent( i, j - 1, k, 0 ) + hCurrent( i, j + 1, k, 0 ) ) / 4;

        
        // Shallow Water Equations
        state_t HxMinus = 0.5 * ( ( hCurrent( i - 1, j, k, 0 ) + hCurrent( i, j, k, 0 ) ) - ( dt ) / ( dx ) * ( ( HXRGFLUXIC ) - ( HXRGFLUXNL ) ) );
        state_t UxMinus = 0.5 * ( ( uCurrent( i - 1, j, k, 0 ) + uCurrent( i, j, k, 0 ) ) - ( dt ) / ( dx ) * ( ( UXRGFLUXIC ) - ( UXRGFLUXNL ) ) );
        state_t VxMinus = 0.5 * ( ( uCurrent( i - 1, j, k, 1 ) + uCurrent( i, j, k, 1 ) ) - ( dt ) / ( dx ) * ( ( VXRGFLUXIC ) - ( VXRGFLUXNL ) ) );

        if ( DEBUG ) printf( "%-10s: %-.4f\t%-10s: %.4f\t%-10s: %.3f\ti: %d\tj:%d\tk: %d\n",  "HxMinus", HxMinus, "UxMinus", UxMinus, "VxMinus", VxMinus, i, j, k );

        state_t HxPlus  = 0.5 * ( ( hCurrent( i, j, k, 0 ) + hCurrent( i + 1, j, k, 0 ) ) - ( dt ) / ( dx ) * ( ( HXRGFLUXNR ) - ( HXRGFLUXIC ) ) );
        state_t UxPlus  = 0.5 * ( ( uCurrent( i, j, k, 0 ) + uCurrent( i + 1, j, k, 0 ) ) - ( dt ) / ( dx ) * ( ( UXRGFLUXNR ) - ( UXRGFLUXIC ) ) );
        state_t VxPlus  = 0.5 * ( ( uCurrent( i, j, k, 1 ) + uCurrent( i + 1, j, k, 1 ) ) - ( dt ) / ( dx ) * ( ( VXRGFLUXNR ) - ( VXRGFLUXIC ) ) );

        if ( DEBUG ) printf( "%-10s: %-.4f\t%-10s: %.4f\t%-10s: %.3f\ti: %d\tj:%d\tk: %d\n",  "HxPlus", HxPlus, "UxPlus", UxPlus, "VxPlus", VxPlus, i, j, k );

        state_t HyMinus = 0.5 * ( ( hCurrent( i, j - 1, k, 0 ) + hCurrent( i, j, k, 0 ) ) - ( dt ) / ( dy ) * ( ( HYRGFLUXIC ) - ( HYRGFLUXNB ) ) );
        state_t UyMinus = 0.5 * ( ( uCurrent( i, j - 1, k, 0 ) + uCurrent( i, j, k, 0 ) ) - ( dt ) / ( dy ) * ( ( UYRGFLUXIC ) - ( UYRGFLUXNB ) ) );
        state_t VyMinus = 0.5 * ( ( uCurrent( i, j - 1, k, 1 ) + uCurrent( i, j, k, 1 ) ) - ( dt ) / ( dy ) * ( ( VYRGFLUXIC ) - ( VYRGFLUXNB ) ) );

        if ( DEBUG )printf( "%-10s: %-.4f\t%-10s: %.4f\t%-10s: %.3f\ti: %d\tj:%d\tk: %d\n",  "HxPlus", HxPlus, "UxPlus", UxPlus, "VxPlus", VxPlus, i, j, k );

        state_t HyPlus  = 0.5 * ( ( hCurrent( i, j, k, 0 ) + hCurrent( i, j + 1, k, 0 ) ) - ( dt ) / ( dy ) * ( ( HYRGFLUXNT ) - ( HYRGFLUXIC ) ) );
        state_t UyPlus  = 0.5 * ( ( uCurrent( i, j, k, 0 ) + uCurrent( i, j + 1, k, 0 ) ) - ( dt ) / ( dy ) * ( ( UYRGFLUXNT ) - ( UYRGFLUXIC ) ) );
        state_t VyPlus  = 0.5 * ( ( uCurrent( i, j, k, 1 ) + uCurrent( i, j + 1, k, 1 ) ) - ( dt ) / ( dy ) * ( ( VYRGFLUXNT ) - ( VYRGFLUXIC ) ) );

        if ( DEBUG )printf( "%-10s: %-.4f\t%-10s: %.4f\t%-10s: %.3f\ti: %d\tj:%d\tk: %d\n",  "HxPlus", HxPlus, "UxPlus", UxPlus, "VxPlus", VxPlus, i, j, k );

        HxFluxMinus( i, j, k, 0 ) = UxMinus;
        UxFluxMinus( i, j, k, 0 ) = ( POW2( UxMinus ) / HxMinus + ghalf * POW2( HxMinus ) );
        UxFluxMinus( i, j, k, 1 ) = UxMinus * VxMinus / HxMinus;

        HxFluxPlus( i, j, k, 0 ) = UxPlus;
        UxFluxPlus( i, j, k, 0 ) = ( POW2( UxPlus ) / HxPlus + ghalf * POW2( HxPlus ) );
        UxFluxPlus( i ,j, k, 1 ) = ( UxPlus * VxPlus / HxPlus );

        HyFluxMinus( i, j, k, 0 ) = VyMinus;
        UyFluxMinus( i, j, k, 0 ) = ( VyMinus * UyMinus / HyMinus );
        UyFluxMinus( i, j, k, 1 ) = ( POW2( VyMinus ) / HyMinus + ghalf * POW2( HyMinus ) );

        HyFluxPlus( i, j, k, 0 ) = VyPlus;
        UyFluxPlus( i, j, k, 0 ) = ( VyPlus * UyPlus / HyPlus );
        UyFluxPlus( i, j, k, 1 ) = ( POW2( VyPlus ) / HyPlus + ghalf * POW2( HyPlus ) );


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

        UWPlus( i, j, k, 1 )   = wCorrector( dt, dy, fabs( VyPlus / HyPlus ) + sqrt( gravity * HyPlus ), uCurrent( i, j + 1, k, 1 ) - uCurrent( i, j, k, 1 ), uCurrent( i, j, k, 1) - uCurrent( i, j - 1, k, 1 ), uCurrent( i, j + 2, k, 1 ) - uCurrent( i, j + 1, k, 1 ) );
        UWPlus( i, j, k, 1 )   *= uCurrent( i, j + 1, k, 1 ) - uCurrent( i, j, k, 1 );


        hNew ( i, j, k, 0 ) = uFullStep( dt, dx, hCurrent( i, j, k, 0 ), HxFluxPlus( i, j, k, 0 ), HxFluxMinus( i, j, k, 0 ), HyFluxPlus( i, j, k, 0 ), HyFluxMinus( i, j, k, 0 ) ) - HxWMinus( i, j, k, 0 ) + HxWPlus( i, j, k, 0 ) - HyWMinus( i, j, k, 0 ) + HyWPlus( i, j, k, 0 );
        uNew ( i, j, k, 0 ) = uFullStep( dt, dx, uCurrent( i, j, k, 0 ), UxFluxPlus( i, j, k, 0 ), UxFluxMinus( i, j, k, 0 ), UyFluxPlus( i, j, k, 0 ), UyFluxMinus( i, j, k, 0 ) ) - UWMinus( i, j, k, 0 ) + UWPlus( i, j, k, 0 );
        uNew ( i, j, k, 1 ) = uFullStep( dt, dy, uCurrent( i, j, k, 1 ), UxFluxPlus( i, j, k, 1 ), UxFluxMinus( i, j, k, 1 ), UyFluxPlus( i, j, k, 1 ), UyFluxMinus( i, j, k, 1 ) ) - UWMinus( i, j, k, 1 ) + UWPlus( i, j, k, 1 );
       
    } );

    MPI_Barrier( MPI_COMM_WORLD );

    haloExchange( pm, a, b );

}

}

}

#endif
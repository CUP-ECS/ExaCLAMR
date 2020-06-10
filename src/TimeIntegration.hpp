/**
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * 
 */

#ifndef EXACLAMR_TIMEINTEGRATION_HPP
#define EXACLAMR_TIMEINTEGRATION_HPP

#ifndef DEBUG
#define DEBUG 0 
#endif

#include <ProblemManager.hpp>

#include <stdio.h>

namespace ExaCLAMR
{
namespace TimeIntegrator
{

#define NEWFIELD( time_step ) ( ( time_step + 1 ) % 2 )
#define CURRENTFIELD( time_step ) ( ( time_step ) % 2 )

template<class ProblemManagerType, class ExecutionSpace, class MemorySpace, typename state_t>
void applyBoundaryConditions( const ProblemManagerType& pm, const ExecutionSpace& exec_space, const MemorySpace& mem_space, const state_t gravity, const int time_step ) {
    if ( pm.mesh()->rank() == 0 && DEBUG ) std::cout << "Applying Boundary Conditions\n";

    using device_type = typename Kokkos::Device<ExecutionSpace, MemorySpace>;

    auto uCurrent = pm.get( Location::Cell(), Field::Velocity(), CURRENTFIELD( time_step ) );
    auto hCurrent = pm.get( Location::Cell(), Field::Height(), CURRENTFIELD( time_step ) );

    auto uNew = pm.get( Location::Cell(), Field::Velocity(), NEWFIELD( time_step ) );
    auto hNew = pm.get( Location::Cell(), Field::Height(), NEWFIELD( time_step ) );

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
            if ( DEBUG ) std::cout << "Rank: " << pm.mesh()->rank() << "\tLeft Boundary:\ti: " << i << "\tj: " << j << "\tk: " << k << "\n";
            hCurrent( i, j, k, 0 ) = hCurrent( i, j + 1, k, 0 );
            uCurrent( i, j, k, 0 ) = uCurrent( i, j + 1, k, 0 );
            uCurrent( i, j, k, 1 ) = -uCurrent( i, j + 1, k, 1 );

            hCurrent( i, j - 1, k, 0 ) = 0;
            uCurrent( i, j - 1, k, 0 ) = 0;
            uCurrent( i, j - 1, k, 1 ) = 0;
        }
        
        if ( j == domain.max( 1 ) && i >= domain.min( 0 ) && i <= domain.max( 0 ) - 1 ) {
            if ( DEBUG ) std::cout << "Rank: " << pm.mesh()->rank() << "\tRight Boundary:\ti: " << i << "\tj: " << j << "\tk: " << k << "\n";
            hCurrent( i, j, k, 0 ) = hCurrent( i, j - 1, k, 0 );
            uCurrent( i, j, k, 0 ) = uCurrent( i, j - 1, k, 0 );
            uCurrent( i, j, k, 1 ) = -uCurrent( i, j - 1, k, 1 );

            hCurrent( i, j + 1, k, 0 ) = 0;
            uCurrent( i, j + 1, k, 0 ) = 0;
            uCurrent( i, j + 1, k, 1 ) = 0;
        }

        if ( i == domain.max( 0 ) && j >= domain.min( 1 ) && j <= domain.max( 1 ) - 1 ) {
            if ( DEBUG ) std::cout << "Rank: " << pm.mesh()->rank() << "\tBottom Boundary:\ti: " << i << "\tj: " << j << "\tk: " << k << "\n";
            hCurrent( i, j, k, 0 ) = hCurrent( i - 1, j, k, 0 );
            uCurrent( i, j, k, 0 ) = -uCurrent( i - 1, j, k, 0 );
            uCurrent( i, j, k, 1 ) = uCurrent( i - 1, j, k, 1 );

            hCurrent( i + 1, j, k, 0 ) = 0;
            uCurrent( i + 1, j, k, 0 ) = 0;
            uCurrent( i + 1, j, k, 1 ) = 0;
        }

        if ( i == domain.min( 0 ) - 1 && j >= domain.min( 1 ) && j <= domain.max( 1 ) - 1 ) {
            if ( DEBUG ) std::cout << "Rank: " << pm.mesh()->rank() << "\tTop Boundary:\ti: " << i << "\tj: " << j << "\tk: " << k << "\n";
            hCurrent( i, j, k, 0 ) = hCurrent( i + 1, j, k, 0 );
            uCurrent( i, j, k, 0 ) = -uCurrent( i + 1, j, k, 0 );
            uCurrent( i, j, k, 1 ) = uCurrent( i + 1, j, k, 1 );

            hCurrent( i - 1, j, k, 0 ) = 0;
            uCurrent( i - 1, j, k, 0 ) = 0;
            uCurrent( i - 1, j, k, 1 ) = 0;
        }
    } );

    Kokkos::fence();
    MPI_Barrier( MPI_COMM_WORLD );

}

template<class ProblemManagerType>
void haloExchange( const ProblemManagerType& pm, const int time_step ) {
    if ( pm.mesh()->rank() == 0 && DEBUG ) std::cout << "Starting Halo Exchange\n";

    pm.gather( Location::Cell(), Field::Velocity(), NEWFIELD( time_step ) );
    pm.gather( Location::Cell(), Field::Height(), NEWFIELD( time_step ) );

    MPI_Barrier( MPI_COMM_WORLD );
}

template<class ProblemManagerType, class ExecutionSpace, class MemorySpace, typename state_t>
state_t setTimeStep( const ProblemManagerType& pm, const ExecutionSpace& exec_space, const MemorySpace& mem_space, const state_t gravity, const state_t sigma, const int time_step ) {

    state_t dx = pm.mesh()->localGrid()->globalGrid().globalMesh().cellSize( 0 );
    state_t dy = pm.mesh()->localGrid()->globalGrid().globalMesh().cellSize( 1 );

    auto uCurrent = pm.get( Location::Cell(), Field::Velocity(), CURRENTFIELD( time_step ) );
    auto hCurrent = pm.get( Location::Cell(), Field::Height(), CURRENTFIELD( time_step ) );

    auto uNew = pm.get( Location::Cell(), Field::Velocity(), NEWFIELD( time_step ) );
    auto hNew = pm.get( Location::Cell(), Field::Height(), NEWFIELD( time_step ) );

    auto domain = pm.mesh()->domainSpace();

    state_t mymindeltaT;

    Kokkos::parallel_reduce( Cajita::createExecutionPolicy( domain, exec_space ), KOKKOS_LAMBDA( const int i, const int j, const int k, state_t& lmin ) {
        state_t wavespeed = sqrt( gravity * hCurrent( i, j, k, 0 ) );
        state_t xspeed = ( fabs( uCurrent( i, j, k, 0 ) + wavespeed ) ) / dx;
        state_t yspeed = ( fabs( uCurrent( i, j, k, 1 ) + wavespeed ) ) / dy;

        state_t deltaT = sigma / ( xspeed + yspeed );

        if ( DEBUG ) std::cout << "'Wavespeed: " << wavespeed << "\txspeed: " << xspeed << "\tyspeed: " << yspeed << "\tdeltaT: " << deltaT << "\n";

        if ( deltaT < lmin ) lmin = deltaT;
    }, Kokkos::Min<state_t>( mymindeltaT ) );

    if ( DEBUG ) std::cout << "dt: " << mymindeltaT << "\n";

    return mymindeltaT;
}

#define POW2( x ) ( ( x ) * ( x ) )

#define HXRGFLUXIC ( uIC )
#define HXRGFLUXNL ( uLeft )
#define HXRGFLUXNR ( uRight )
#define HXRGFLUXNT ( uBot )
#define HXRGFLUXNB ( uTop )

#define UXRGFLUXIC ( POW2( uIC )    / hIC    + ghalf * POW2( hIC ) )
#define UXRGFLUXNL ( POW2( uLeft )  / hLeft  + ghalf * POW2( hLeft ) )
#define UXRGFLUXNR ( POW2( uRight ) / hRight + ghalf * POW2( hRight ) )
#define UXRGFLUXNB ( POW2( uBot )   / hBot   + ghalf * POW2( hBot ) )
#define UXRGFLUXNT ( POW2( uTop )   / hTop   + ghalf * POW2( hTop ) )

#define VXRGFLUXIC ( uIC    * vIC    / hIC )
#define VXRGFLUXNL ( uLeft  * vLeft  / hLeft )
#define VXRGFLUXNR ( uRight * vRight / hRight )
#define VXRGFLUXNB ( uBot   * vBot   / hBot )
#define VXRGFLUXNT ( uTop   * vTop   / hTop )

#define HYRGFLUXIC ( vIC )
#define HYRGFLUXNL ( vLeft )
#define HYRGFLUXNR ( vRight )
#define HYRGFLUXNB ( vBot )
#define HYRGFLUXNT ( vTop )

#define UYRGFLUXIC ( vIC    * uIC    / hIC )
#define UYRGFLUXNL ( vLeft  * uLeft  / hLeft )
#define UYRGFLUXNR ( vRight * uRight / hRight )
#define UYRGFLUXNB ( vBot   * uBot   / hBot )
#define UYRGFLUXNT ( vTop   * uTop   / hTop )

#define VYRGFLUXIC ( POW2( vIC )    / hIC    + ghalf * POW2( hIC ) )
#define VYRGFLUXNL ( POW2( vLeft )  / hLeft  + ghalf * POW2( hLeft ) )
#define VYRGFLUXNR ( POW2( vRight ) / hRight + ghalf * POW2( hRight ) )
#define VYRGFLUXNB ( POW2( vBot )   / hBot   + ghalf * POW2( hBot ) )
#define VYRGFLUXNT ( POW2( vTop )   / hTop   + ghalf * POW2( hTop ) )

template<typename state_t>
inline state_t wCorrector( state_t dt, state_t dr, state_t uEigen, state_t gradHalf, state_t gradMinus, state_t gradPlus ) {
    state_t nu = 0.5 * uEigen * dt / dr;
    nu *= ( 1.0 - nu );

    state_t rDenom = 1.0 / std::max( POW2( gradHalf ), 1.0e-30 );
    state_t rPlus = ( gradPlus * gradHalf ) * rDenom;
    state_t rMinus = ( gradMinus * gradHalf ) * rDenom;

    return 0.5 * nu * ( 1.0 - std::max( std::min( std::min( 1.0, rPlus ), rMinus ), 0.0 ) );
}

template<typename state_t>
inline state_t uFullStep( state_t dt, state_t dr, state_t U, state_t FPlus, state_t FMinus, state_t GPlus, state_t GMinus ) {
    return ( U + ( - ( dt / dr) * ( ( FPlus - FMinus ) + ( GPlus - GMinus ) ) ) );
}

template<class ProblemManagerType, class ExecutionSpace, class MemorySpace, typename state_t>
void step( const ProblemManagerType& pm, const ExecutionSpace& exec_space, const MemorySpace& mem_space, const state_t dt, const state_t gravity, const int time_step ) {
    if ( pm.mesh()->rank() == 0 && DEBUG ) std::cout << "Time Stepper\n";

    using device_type = typename Kokkos::Device<ExecutionSpace, MemorySpace>;

    state_t dx = pm.mesh()->localGrid()->globalGrid().globalMesh().cellSize( 0 );
    state_t dy = pm.mesh()->localGrid()->globalGrid().globalMesh().cellSize( 1 );
    state_t ghalf = 0.5 * gravity;

    applyBoundaryConditions( pm, exec_space, mem_space, gravity, time_step );

    auto uCurrent = pm.get( Location::Cell(), Field::Velocity(), CURRENTFIELD( time_step ) );
    auto hCurrent = pm.get( Location::Cell(), Field::Height(), CURRENTFIELD( time_step ) );

    auto uNew = pm.get( Location::Cell(), Field::Velocity(), NEWFIELD( time_step ) );
    auto hNew = pm.get( Location::Cell(), Field::Height(), NEWFIELD( time_step ) );

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

    if ( DEBUG ) std::cout << "Domain Space: " << domainSpace.min( 0 ) << domainSpace.min( 1 ) << domainSpace.min( 2 ) << \
    domainSpace.max( 0 ) << domainSpace.max( 1 ) << domainSpace.max( 2 ) << "\n";

    Kokkos::parallel_for( Cajita::createExecutionPolicy( domainSpace, exec_space ), KOKKOS_LAMBDA( const int i, const int j, const int k ) {            
        if ( DEBUG ) {
            auto local_mesh = Cajita::createLocalMesh<device_type>( * pm.mesh()->localGrid() );
            int coords[3] = { i, j, k };
            state_t x[3];
            local_mesh.coordinates( Cajita::Cell(), coords, x );
            std::cout << "Rank: " << pm.mesh()->rank() << "\ti: " << i << "\tj: " << j << "\tk: " << k << "\tx: " << x[0] << "\ty: " << x[1] << "\tz: " << x[2] << "\n";
        }
            
        // Simple Diffusion Problem
        // hNew( i, j, k, 0 ) = ( hCurrent( i - 1, j, k, 0 ) + hCurrent( i + 1, j, k, 0 ) + hCurrent( i, j - 1, k, 0 ) + hCurrent( i, j + 1, k, 0 ) ) / 4;

        //
        state_t hIC     = hCurrent( i, j, k, 0 );
        state_t hLeft   = hCurrent( i - 1, j, k, 0 );
        state_t hRight  = hCurrent( i + 1, j, k, 0 );
        state_t hBot    = hCurrent( i, j - 1, k, 0 );
        state_t hTop    = hCurrent( i, j + 1, k, 0 );
        state_t hLeft2  = hCurrent( i - 2, j, k, 0 );
        state_t hRight2 = hCurrent( i + 2, j, k, 0 );
        state_t hBot2   = hCurrent( i, j - 2, k, 0 );
        state_t hTop2   = hCurrent( i, j + 2, k, 0 );

        state_t uIC     = uCurrent( i, j, k, 0 );
        state_t uLeft   = uCurrent( i - 1, j, k, 0 );
        state_t uRight  = uCurrent( i + 1, j, k, 0 );
        state_t uBot    = uCurrent( i, j - 1, k, 0 );
        state_t uTop    = uCurrent( i, j + 1, k, 0 );
        state_t uLeft2  = uCurrent( i - 2, j, k, 0 );
        state_t uRight2 = uCurrent( i + 2, j, k, 0 );
        state_t uBot2   = uCurrent( i, j - 2, k, 0 );
        state_t uTop2   = uCurrent( i, j + 2, k, 0 );

        state_t vIC     = uCurrent( i, j, k, 1 );
        state_t vLeft   = uCurrent( i - 1, j, k, 1 );
        state_t vRight  = uCurrent( i + 1, j, k, 1 );
        state_t vBot    = uCurrent( i, j - 1, k, 1 );
        state_t vTop    = uCurrent( i, j + 1, k, 1 );
        state_t vLeft2  = uCurrent( i - 2, j, k, 1 );
        state_t vRight2 = uCurrent( i + 2, j, k, 1 );
        state_t vBot2   = uCurrent( i, j - 2, k, 1 );
        state_t vTop2   = uCurrent( i, j + 2, k, 1 );

        // Shallow Water Equations
        state_t HxMinus = 0.5 * ( ( hLeft + hIC ) - ( dt ) / ( dx ) * ( ( HXRGFLUXIC ) - ( HXRGFLUXNL ) ) );
        state_t UxMinus = 0.5 * ( ( uLeft + uIC ) - ( dt ) / ( dx ) * ( ( UXRGFLUXIC ) - ( UXRGFLUXNL ) ) );
        state_t VxMinus = 0.5 * ( ( vLeft + vIC ) - ( dt ) / ( dx ) * ( ( VXRGFLUXIC ) - ( VXRGFLUXNL ) ) );

        if ( DEBUG ) std::cout << std::left << std::setw(10) << "HxMinus: " << std::setw(6) << HxMinus << \
        "\tUxMinus: " << std::setw(6) << UxMinus << "\tVxMinus: " << std::setw(6) << VxMinus << "\ti: " << i << "\tj: "<< j << "\tk: " << k << "\n";

        state_t HxPlus  = 0.5 * ( ( hIC + hRight ) - ( dt ) / ( dx ) * ( ( HXRGFLUXNR ) - ( HXRGFLUXIC ) ) );
        state_t UxPlus  = 0.5 * ( ( uIC + uRight ) - ( dt ) / ( dx ) * ( ( UXRGFLUXNR ) - ( UXRGFLUXIC ) ) );
        state_t VxPlus  = 0.5 * ( ( vIC + vRight ) - ( dt ) / ( dx ) * ( ( VXRGFLUXNR ) - ( VXRGFLUXIC ) ) );

        if ( DEBUG ) std::cout << std::left << std::setw(10) << "HxPlus: " << std::setw(6) << HxPlus << \
        "\tUxPlus: " << std::setw(6) << UxPlus << "\tVxPlus: " << std::setw(6) << VxPlus << "\ti: " << i << "\tj: " << j << "\tk: " << k << "\n";

        state_t HyMinus = 0.5 * ( ( hBot + hIC ) - ( dt ) / ( dy ) * ( ( HYRGFLUXIC ) - ( HYRGFLUXNB ) ) );
        state_t UyMinus = 0.5 * ( ( uBot + uIC ) - ( dt ) / ( dy ) * ( ( UYRGFLUXIC ) - ( UYRGFLUXNB ) ) );
        state_t VyMinus = 0.5 * ( ( vBot + vIC ) - ( dt ) / ( dy ) * ( ( VYRGFLUXIC ) - ( VYRGFLUXNB ) ) );

        if ( DEBUG ) std::cout << std::left << std::setw(10) << "HyMinus: " << std::setw(6) << HyMinus << \
        "\tUyMinus: " << std::setw(6) << UyMinus << "\tVyMinus: " << std::setw(6) << VyMinus << "\ti: " << i << "\tj: " << j << "\tk: " << k << "\n";

        state_t HyPlus  = 0.5 * ( ( hIC + hTop ) - ( dt ) / ( dy ) * ( ( HYRGFLUXNT ) - ( HYRGFLUXIC ) ) );
        state_t UyPlus  = 0.5 * ( ( uIC + uTop ) - ( dt ) / ( dy ) * ( ( UYRGFLUXNT ) - ( UYRGFLUXIC ) ) );
        state_t VyPlus  = 0.5 * ( ( vIC + vTop ) - ( dt ) / ( dy ) * ( ( VYRGFLUXNT ) - ( VYRGFLUXIC ) ) );

        if ( DEBUG ) std::cout << std::left << std::setw(10) << "HyPlus: " << std::setw(6) << HyPlus << \
        "\tUyPlus: " << std::setw(6) << UyPlus << "\tVyPlus: " << std::setw(6) << VyPlus << "\ti: " << i << "\tj: " << j << "\tk: " << k << "\n";

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


        HxWMinus( i, j, k, 0 ) = wCorrector( dt, dx, fabs( UxMinus / HxMinus ) + sqrt( gravity * HxMinus ), hIC - hLeft, hLeft - hLeft2, hRight - hIC );
        HxWMinus( i, j, k, 0 ) *= hIC - hLeft;
        
        HxWPlus( i, j, k, 0 )  = wCorrector( dt, dx, fabs( UxPlus / HxPlus ) + sqrt( gravity * HxPlus ), hRight - hIC, hIC - hLeft, hRight2 - hRight );
        HxWPlus( i, j, k, 0 )  *= hRight - hIC;

        UWMinus( i, j, k, 0 )  = wCorrector( dt, dx, fabs( UxMinus / HxMinus ) + sqrt( gravity * HxMinus ), uIC - uLeft, uLeft - uLeft2, uRight - uIC );
        UWMinus( i, j, k, 0 )  *= uIC - uLeft;

        UWPlus( i, j, k, 0 )   = wCorrector( dt, dx, fabs( UxPlus / HxPlus ) + sqrt( gravity * HxPlus ), uRight - uIC, uIC - uLeft, uRight2 - uRight );
        UWPlus( i, j, k, 0 )   *= uRight - uIC;


        HyWMinus( i, j, k, 0 ) = wCorrector( dt, dy, fabs( VyMinus / HyMinus ) + sqrt( gravity * HyMinus ), hIC - hBot, hBot - hBot2, hTop - hIC );
        HyWMinus( i, j, k, 0 ) *= hIC - hBot;

        HyWPlus( i, j, k, 0 )  = wCorrector( dt, dy, fabs( VyPlus / HyPlus ) + sqrt( gravity * HyPlus ), hTop - hIC, hIC - hBot, hTop2 - hTop );
        HyWPlus( i, j, k, 0 )  *= hTop - hIC;

        UWMinus( i, j, k, 1 )  = wCorrector( dt, dy, fabs( VyMinus / HyMinus ) + sqrt( gravity * HyMinus ), vIC - vBot, vBot - vBot2, vTop - vIC );
        UWMinus( i, j, k, 1 )  *= vIC - vBot;

        UWPlus( i, j, k, 1 )   = wCorrector( dt, dy, fabs( VyPlus / HyPlus ) + sqrt( gravity * HyPlus ), vTop - vIC, vIC - vBot, vTop2 - vTop );
        UWPlus( i, j, k, 1 )   *= vTop - vIC;


        hNew ( i, j, k, 0 ) = uFullStep( dt, dx, hIC, HxFluxPlus( i, j, k, 0 ), HxFluxMinus( i, j, k, 0 ), HyFluxPlus( i, j, k, 0 ), HyFluxMinus( i, j, k, 0 ) ) - HxWMinus( i, j, k, 0 ) + HxWPlus( i, j, k, 0 ) - HyWMinus( i, j, k, 0 ) + HyWPlus( i, j, k, 0 );
        uNew ( i, j, k, 0 ) = uFullStep( dt, dx, uIC, UxFluxPlus( i, j, k, 0 ), UxFluxMinus( i, j, k, 0 ), UyFluxPlus( i, j, k, 0 ), UyFluxMinus( i, j, k, 0 ) ) - UWMinus( i, j, k, 0 ) + UWPlus( i, j, k, 0 );
        uNew ( i, j, k, 1 ) = uFullStep( dt, dy, vIC, UxFluxPlus( i, j, k, 1 ), UxFluxMinus( i, j, k, 1 ), UyFluxPlus( i, j, k, 1 ), UyFluxMinus( i, j, k, 1 ) ) - UWMinus( i, j, k, 1 ) + UWPlus( i, j, k, 1 );
       
    } );

    Kokkos::fence();
    MPI_Barrier( MPI_COMM_WORLD );

}

}

}

#endif
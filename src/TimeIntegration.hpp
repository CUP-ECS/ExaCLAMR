/**
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * Time Integration Step, include functions to:
 * Apply boundary conditions
 * Perform halo exchange
 * Calculate dynamic timestep based on wave speed
 * Flux corrector calculation
 * Full time step calculation
 * Integration step using the shallow water equations
 */

#ifndef EXACLAMR_TIMEINTEGRATION_HPP
#define EXACLAMR_TIMEINTEGRATION_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <BoundaryConditions.hpp>
#include <ExaCLAMR.hpp>
#include <ProblemManager.hpp>

#include <math.h>
#include <stdio.h>

namespace ExaCLAMR {
    namespace TimeIntegrator {

/**
* Perform Halo Exchange
*
* @param pm Problem manager
* @param exec_space Execution space
* @param mem_space Memory space
* @param mindt Time increment/step (dt)
* @param time_step Current time step
**/
        template <typename state_t, class ProblemManagerType, class ExecutionSpace>
        void haloExchange( const ProblemManagerType &pm, const ExecutionSpace &exec_space, const int time_step ) {
            // DEBUG: Trace in Halo Exchange
            if ( pm.mesh()->rank() == 0 && DEBUG ) std::cout << "Starting Halo Exchange\n";
            if ( pm.mesh()->rank() == 0 && DEBUG ) std::cout << exec_space.name() << "\n";

            // Perform Halo Exchange on Height and Momentum State Views
            pm.gather( Location::Cell(), NEWFIELD( time_step ) );
        }

/**
* Calculate dynamic time step based off of wave speed and cell size
*
* @param pm Problem manager
* @param exec_space Execution space
* @param mem_space Memory space
* @param gravity Gravitational constant
* @param sigma Factor to control CFL number, stability and size of time step
* @param time_step Current time step
**/
        template <class ProblemManagerType, class ExecutionSpace, typename state_t>
        state_t setTimeStep( const ProblemManagerType &pm, const ExecutionSpace &exec_space, const state_t gravity, const state_t sigma, const int time_step ) {
            // Get dx and dy of Regular Mesh Cell
            state_t dx = pm.mesh()->cellSize( 0 );
            state_t dy = pm.mesh()->cellSize( 1 );

            // Get Current State Variables
            auto h_current = pm.get( Location::Cell(), Field::Height(), CURRENTFIELD( time_step ) );
            auto u_current = pm.get( Location::Cell(), Field::Momentum(), CURRENTFIELD( time_step ) );

            // Get Domain Index Space to Loop Over
            auto domain = pm.mesh()->domainSpace();

            // Initialize Overall Minimum Time Step
            state_t dt_min;

            // Kokkos Parallel Reduce of Domain Index Space to Calculate Time Step ( i, j, k )
            Kokkos::parallel_reduce(
                "Set_TimeStep", Cajita::createExecutionPolicy( domain, exec_space ), KOKKOS_LAMBDA( const int i, const int j, const int k, state_t &lmin ) {
                    // Wave Speed Calculation
                    state_t wavespeed = sqrt( gravity * h_current( i, j, k, 0 ) );
                    state_t xspeed    = ( fabs( u_current( i, j, k, 0 ) + wavespeed ) ) / dx;
                    state_t yspeed    = ( fabs( u_current( i, j, k, 1 ) + wavespeed ) ) / dy;

                    // Time Step Calculation
                    state_t dt = sigma / ( xspeed + yspeed );

                    // DEBUG: Print Wave Speed, X Speed, Y Speed, and Time Step Calculated for Current Index ( i, j, k )
                    // if ( DEBUG ) std::cout << "'Wavespeed: " << wavespeed << "\txspeed: " << xspeed << "\tyspeed: " << yspeed << "\tdeltaT: " << dt << "\n";

                    // Set Minimum
                    if ( dt < lmin ) lmin = dt;
                },
                Kokkos::Min<state_t>( dt_min ) );

            // DEBUG: Print Overall Minimum Time Step
            if ( DEBUG ) std::cout << "dt: " << dt_min << "\n";

            return dt_min;
        }

/**
 * Flux for Uy and Vx directions
 * 
 * @param u X-Momentum ( u )
 * @param v Y-Momentum ( v )
 * @param h Height
**/
        template <typename state_t>
        KOKKOS_INLINE_FUNCTION
            state_t
            fluxUyVx( state_t u, state_t v, state_t h ) {
            return u * v / h;
        }

/**
 * Flux for Ux and Vy directions
 * 
 * @param u Momentum (x or y i.e. u or v)
 * @param h Height
 * @param ghalf Half of the gravitational constant
**/
        template <typename state_t>
        KOKKOS_INLINE_FUNCTION
            state_t
            fluxUxVy( state_t u, state_t h, state_t ghalf ) {
            return POW2( u ) / h + ghalf * POW2( h );
        }

/**
 * Flux corrector
 * 
 * @param dt Time step
 * @param dr Spatial step (dx or dy)
 * @param u_eigen Eigenvalue of the system
 * @param grad_half Gradient
 * @param grad_minus Gradient to the left/bottom
 * @param grad_plus Gradient to the right/top 
**/
        template <typename state_t>
        KOKKOS_INLINE_FUNCTION
            state_t
            wCorrector( state_t dt, state_t dr, state_t u_eigen, state_t grad_half, state_t grad_minus, state_t grad_plus ) {
            state_t nu = 0.5 * u_eigen * dt / dr;
            nu *= ( 1.0 - nu );

            state_t rDenom = 1.0 / fmax( POW2( grad_half ), 1.0e-30 );
            state_t rPlus  = ( grad_plus * grad_half ) * rDenom;
            state_t rMinus = ( grad_minus * grad_half ) * rDenom;

            return 0.5 * nu * ( 1.0 - fmax( fmin( fmin( 1.0, rPlus ), rMinus ), 0.0 ) );
        }

/**
 * Full Step Shallow Water Calculation
 * 
 * @param dt Time step
 * @param dr Spatial step (dx or dy)
 * @param u Current state value
 * @param f_plus Positive flux in x-direction
 * @param f_minus Negative flux in x-direction
 * @param g_plus Positive flux in y-direction
 * @param g_minus Negative flux in y-direction
**/
        template <typename state_t>
        KOKKOS_INLINE_FUNCTION
            state_t
            uFullStep( state_t dt, state_t dr, state_t u, state_t f_plus, state_t f_minus, state_t g_plus, state_t g_minus ) {
            return ( u + ( -( dt / dr ) * ( ( f_plus - f_minus ) + ( g_plus - g_minus ) ) ) );
        }

/**
 * Time Step Iteration of Shallow Water Equations
 * 
 * @param pm Problem manager
 * @param exec_space Execution space
 * @param mem_space Memory space
 * @param bc Boundary conditions
 * @param dt Time step (dt)
 * @param gravity Gravitational constant
 * @param time_step Current time step (count) 
**/
        template <class ProblemManagerType, class ExecutionSpace, typename state_t>
        void step( const ProblemManagerType &pm, const ExecutionSpace &exec_space, const ExaCLAMR::BoundaryCondition &bc, const state_t dt, const state_t gravity, const int time_step ) {
            if ( pm.mesh()->rank() == 0 && DEBUG ) std::cout << "Time Stepper\n";

            // Get dx and dy
            state_t dx    = pm.mesh()->cellSize( 0 );
            state_t dy    = pm.mesh()->cellSize( 1 );
            state_t ghalf = 0.5 * gravity;

            // Get Local Grid to get Owned Index Space
            auto local_grid  = pm.mesh()->localGrid();
            auto owned_cells = local_grid->indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );

            // Get Domain Index Space to Check Boundaries
            auto mesh = *pm.mesh();

            // Get Current State Views
            auto u_current = pm.get( Location::Cell(), Field::Momentum(), CURRENTFIELD( time_step ) );
            auto h_current = pm.get( Location::Cell(), Field::Height(), CURRENTFIELD( time_step ) );

            // Apply Boundary Conditions
            // DEBUG: Print Boundary Condition Trace
            if ( pm.mesh()->rank() == 0 && DEBUG ) std::cout << "Applying Boundary Conditions\n";
            // Loop Over All Owned Cells and Update Boundary Cells ( i, j, k )
            Kokkos::parallel_for(
                "Boundary_Conditions", Cajita::createExecutionPolicy( owned_cells, exec_space ), KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                    bc( pm, i, j, k, h_current, u_current, mesh );
                } );

            // Kokkos Fence
            Kokkos::fence();

            // Get New State Views
            auto u_new = pm.get( Location::Cell(), Field::Momentum(), NEWFIELD( time_step ) );
            auto h_new = pm.get( Location::Cell(), Field::Height(), NEWFIELD( time_step ) );

            // Get Flux Views
            auto hx_flux_plus  = pm.get( Location::Cell(), Field::HxFluxPlus() );
            auto hx_flux_minus = pm.get( Location::Cell(), Field::HxFluxMinus() );
            auto ux_flux_plus  = pm.get( Location::Cell(), Field::UxFluxPlus() );
            auto ux_flux_minus = pm.get( Location::Cell(), Field::UxFluxMinus() );

            auto hy_flux_plus  = pm.get( Location::Cell(), Field::HyFluxPlus() );
            auto hy_flux_minus = pm.get( Location::Cell(), Field::HyFluxMinus() );
            auto uy_flux_plus  = pm.get( Location::Cell(), Field::UyFluxPlus() );
            auto uy_flux_minus = pm.get( Location::Cell(), Field::UyFluxMinus() );

            // Get Flux Corrector Views
            auto hx_w_plus  = pm.get( Location::Cell(), Field::HxWPlus() );
            auto hx_w_minus = pm.get( Location::Cell(), Field::HxWMinus() );
            auto hy_w_plus  = pm.get( Location::Cell(), Field::HyWPlus() );
            auto hy_w_minus = pm.get( Location::Cell(), Field::HyWMinus() );

            auto u_w_plus  = pm.get( Location::Cell(), Field::UWPlus() );
            auto u_w_minus = pm.get( Location::Cell(), Field::UWMinus() );

            auto domain = pm.mesh()->domainSpace();

            // DEBUG: Print out Domain Space Indices
            if ( DEBUG ) std::cout << "Domain Space: " << domain.min( 0 ) << domain.min( 1 ) << domain.min( 2 ) << domain.max( 0 ) << domain.max( 1 ) << domain.max( 2 ) << "\n";

            // Kokkos Parallel Section over Domain Space Indices to Calculate New State Values ( i, j, k )
            Kokkos::parallel_for(
                "Finite_Volume", Cajita::createExecutionPolicy( domain, exec_space ), KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                    // Simple Diffusion Problem
                    // h_new( i, j, k, 0 ) = ( h_current( i - 1, j, k, 0 ) + h_current( i + 1, j, k, 0 ) + h_current( i, j - 1, k, 0 ) + h_current( i, j + 1, k, 0 ) ) / 4;

                    // Store Current Iteration Values Locally to Speed Up Performance and Reduce Fetches and Cache Misses
                    state_t h_ic     = h_current( i, j, k, 0 );
                    state_t h_left   = h_current( i - 1, j, k, 0 );
                    state_t h_right  = h_current( i + 1, j, k, 0 );
                    state_t h_bot    = h_current( i, j - 1, k, 0 );
                    state_t h_top    = h_current( i, j + 1, k, 0 );
                    state_t h_left2  = h_current( i - 2, j, k, 0 );
                    state_t h_right2 = h_current( i + 2, j, k, 0 );
                    state_t h_bot2   = h_current( i, j - 2, k, 0 );
                    state_t h_top2   = h_current( i, j + 2, k, 0 );

                    state_t u_ic     = u_current( i, j, k, 0 );
                    state_t u_left   = u_current( i - 1, j, k, 0 );
                    state_t u_right  = u_current( i + 1, j, k, 0 );
                    state_t u_bot    = u_current( i, j - 1, k, 0 );
                    state_t u_top    = u_current( i, j + 1, k, 0 );
                    state_t u_left2  = u_current( i - 2, j, k, 0 );
                    state_t u_right2 = u_current( i + 2, j, k, 0 );
                    state_t u_bot2   = u_current( i, j - 2, k, 0 );
                    state_t u_top2   = u_current( i, j + 2, k, 0 );

                    state_t v_ic     = u_current( i, j, k, 1 );
                    state_t v_left   = u_current( i - 1, j, k, 1 );
                    state_t v_right  = u_current( i + 1, j, k, 1 );
                    state_t v_bot    = u_current( i, j - 1, k, 1 );
                    state_t v_top    = u_current( i, j + 1, k, 1 );
                    state_t v_left2  = u_current( i - 2, j, k, 1 );
                    state_t v_right2 = u_current( i + 2, j, k, 1 );
                    state_t v_bot2   = u_current( i, j - 2, k, 1 );
                    state_t v_top2   = u_current( i, j + 2, k, 1 );

                    // Shallow Water Equations
                    // X Minus Direction
                    state_t hx_minus = 0.5 * ( ( h_left + h_ic ) - ( dt ) / ( dx ) * ( ( u_ic ) - ( u_left ) ) );
                    state_t ux_minus = 0.5 * ( ( u_left + u_ic ) - ( dt ) / ( dx ) * ( ( fluxUxVy( u_ic, h_ic, ghalf ) ) - ( fluxUxVy( u_left, h_left, ghalf ) ) ) );
                    state_t vx_minus = 0.5 * ( ( v_left + v_ic ) - ( dt ) / ( dx ) * ( ( fluxUyVx( u_ic, v_ic, h_ic ) ) - ( fluxUyVx( u_left, v_left, h_left ) ) ) );

                    // DEBUG: Print hx_minus, ux_minus, vx_minus, i, j, k
                    // if ( DEBUG ) std::cout << std::left << std::setw( 10 ) << "hx_minus: " << std::setw( 6 ) << hx_minus << \
                    "\tux_minus: " << std::setw( 6 ) << ux_minus << "\tvx_minus: " << std::setw( 6 ) << vx_minus << "\ti: " << i << "\tj: "<< j << "\tk: " << k << "\n";

                    // X Plus Direction
                    state_t hx_plus = 0.5 * ( ( h_ic + h_right ) - ( dt ) / ( dx ) * ( ( u_right ) - ( u_ic ) ) );
                    state_t ux_plus = 0.5 * ( ( u_ic + u_right ) - ( dt ) / ( dx ) * ( ( fluxUxVy( u_right, h_right, ghalf ) ) - ( fluxUxVy( u_ic, h_ic, ghalf ) ) ) );
                    state_t vx_plus = 0.5 * ( ( v_ic + v_right ) - ( dt ) / ( dx ) * ( ( fluxUyVx( u_right, v_right, h_right ) ) - ( fluxUyVx( u_ic, v_ic, h_ic ) ) ) );

                    // DEBUG: Print hx_plus, ux_plus, vx_plus, i, j, k
                    // if ( DEBUG ) std::cout << std::left << std::setw( 10 ) << "hx_plus: " << std::setw( 6 ) << hx_plus << \
                    "\tux_plus: " << std::setw( 6 ) << ux_plus << "\tvx_plus: " << std::setw( 6 ) << vx_plus << "\ti: " << i << "\tj: " << j << "\tk: " << k << "\n";

                    // Y Minus Direction
                    state_t hy_minus = 0.5 * ( ( h_bot + h_ic ) - ( dt ) / ( dy ) * ( ( v_ic ) - ( v_bot ) ) );
                    state_t uy_minus = 0.5 * ( ( u_bot + u_ic ) - ( dt ) / ( dy ) * ( ( fluxUyVx( u_ic, v_ic, h_ic ) ) - ( fluxUyVx( u_bot, v_bot, h_bot ) ) ) );
                    state_t vy_minus = 0.5 * ( ( v_bot + v_ic ) - ( dt ) / ( dy ) * ( ( fluxUxVy( v_ic, h_ic, ghalf ) ) - ( fluxUxVy( v_bot, h_bot, ghalf ) ) ) );

                    // DEBUG: Print hy_minus, uy_minus, vy_minus, i, j, k
                    // if ( DEBUG ) std::cout << std::left << std::setw( 10 ) << "hy_minus: " << std::setw( 6 ) << hy_minus << \
                    "\tuy_minus: " << std::setw( 6 ) << uy_minus << "\tvy_minus: " << std::setw( 6 ) << vy_minus << "\ti: " << i << "\tj: " << j << "\tk: " << k << "\n";

                    // Y Plus Direction
                    state_t hy_plus = 0.5 * ( ( h_ic + h_top ) - ( dt ) / ( dy ) * ( ( v_top ) - ( v_ic ) ) );
                    state_t uy_plus = 0.5 * ( ( u_ic + u_top ) - ( dt ) / ( dy ) * ( ( fluxUyVx( u_top, v_top, h_top ) ) - ( fluxUyVx( u_ic, v_ic, h_ic ) ) ) );
                    state_t vy_plus = 0.5 * ( ( v_ic + v_top ) - ( dt ) / ( dy ) * ( ( fluxUxVy( v_top, h_top, ghalf ) ) - ( fluxUxVy( v_ic, h_ic, ghalf ) ) ) );

                    // DEBUG: Print hy_plus, uy_plus, vy_plus, i, j, k
                    // if ( DEBUG ) std::cout << std::left << std::setw( 10 ) << "hy_plus: " << std::setw( 6 ) << hy_plus << \
                    "\tuy_plus: " << std::setw( 6 ) << uy_plus << "\tvy_plus: " << std::setw( 6 ) << vy_plus << "\ti: " << i << "\tj: " << j << "\tk: " << k << "\n";

                    // Flux View Updates
                    // X Direction
                    hx_flux_minus( i, j, k, 0 ) = ux_minus;
                    ux_flux_minus( i, j, k, 0 ) = ( POW2( ux_minus ) / hx_minus + ghalf * POW2( hx_minus ) );
                    ux_flux_minus( i, j, k, 1 ) = ux_minus * vx_minus / hx_minus;

                    // DEBUG: Print hx_flux_minus, ux_flux_minus, ux_flux_minus, i, j, k
                    // if ( DEBUG ) std::cout << std::left << std::setw( 10 ) << "hx_flux_minus: " << std::setw( 6 ) << hx_flux_minus( i, j, k, 0 ) << \
                    "\tux_flux_minus: " << std::setw( 6 ) << ux_flux_minus( i, j, k, 0 ) << "\tux_flux_minus: " << std::setw( 6 ) << ux_flux_minus( i, j, k, 1 ) << \
                    "\ti: " << i << "\tj: " << j << "\tk: " << k << "\n";

                    hx_flux_plus( i, j, k, 0 ) = ux_plus;
                    ux_flux_plus( i, j, k, 0 ) = ( POW2( ux_plus ) / hx_plus + ghalf * POW2( hx_plus ) );
                    ux_flux_plus( i, j, k, 1 ) = ( ux_plus * vx_plus / hx_plus );

                    // DEBUG: Print hx_flux_plus, ux_flux_plus, ux_flux_plus, i, j, k
                    // if ( DEBUG ) std::cout << std::left << std::setw( 10 ) << "hx_flux_plus: " << std::setw( 6 ) << hx_flux_plus( i, j, k, 0 ) << \
                    "\tux_flux_plus: " << std::setw( 6 ) << ux_flux_plus( i, j, k, 0 ) << "\tux_flux_plus: " << std::setw( 6 ) << ux_flux_plus( i, j, k, 1 ) << \
                    "\ti: " << i << "\tj: " << j << "\tk: " << k << "\n";

                    // Y Direction
                    hy_flux_minus( i, j, k, 0 ) = vy_minus;
                    uy_flux_minus( i, j, k, 0 ) = ( vy_minus * uy_minus / hy_minus );
                    uy_flux_minus( i, j, k, 1 ) = ( POW2( vy_minus ) / hy_minus + ghalf * POW2( hy_minus ) );

                    // DEBUG: Print hy_flux_minus, uy_flux_minus, uy_flux_minus, i, j, k
                    // if ( DEBUG ) std::cout << std::left << std::setw( 10 ) << "hy_flux_minus: " << std::setw( 6 ) << hy_flux_minus( i, j, k, 0 ) << \
                    "\tuy_flux_minus: " << std::setw( 6 ) << uy_flux_minus( i, j, k, 0 ) << "\tuy_flux_minus: " << std::setw( 6 ) << uy_flux_minus( i, j, k, 1 ) << \
                    "\ti: " << i << "\tj: " << j << "\tk: " << k << "\n";

                    hy_flux_plus( i, j, k, 0 ) = vy_plus;
                    uy_flux_plus( i, j, k, 0 ) = ( vy_plus * uy_plus / hy_plus );
                    uy_flux_plus( i, j, k, 1 ) = ( POW2( vy_plus ) / hy_plus + ghalf * POW2( hy_plus ) );

                    // DEBUG: Print hy_flux_plus, uy_flux_plus, uy_flux_plus, i, j, k
                    // if ( DEBUG ) std::cout << std::left << std::setw( 10 ) << "hy_flux_plus: " << std::setw( 6 ) << hy_flux_plus( i, j, k, 0 ) << \
                    "\tuy_flux_plus: " << std::setw( 6 ) << uy_flux_plus( i, j, k, 0 ) << "\tuy_flux_plus: " << std::setw( 6 ) << uy_flux_plus( i, j, k, 1 ) << \
                    "\ti: " << i << "\tj: " << j << "\tk: " << k << "\n";

                    // Flux Corrector Calculations
                    // X Direction
                    hx_w_minus( i, j, k, 0 ) = wCorrector( dt, dx, fabs( ux_minus / hx_minus ) + sqrt( gravity * hx_minus ), h_ic - h_left, h_left - h_left2, h_right - h_ic );
                    hx_w_minus( i, j, k, 0 ) *= h_ic - h_left;

                    hx_w_plus( i, j, k, 0 ) = wCorrector( dt, dx, fabs( ux_plus / hx_plus ) + sqrt( gravity * hx_plus ), h_right - h_ic, h_ic - h_left, h_right2 - h_right );
                    hx_w_plus( i, j, k, 0 ) *= h_right - h_ic;

                    u_w_minus( i, j, k, 0 ) = wCorrector( dt, dx, fabs( ux_minus / hx_minus ) + sqrt( gravity * hx_minus ), u_ic - u_left, u_left - u_left2, u_right - u_ic );
                    u_w_minus( i, j, k, 0 ) *= u_ic - u_left;

                    u_w_plus( i, j, k, 0 ) = wCorrector( dt, dx, fabs( ux_plus / hx_plus ) + sqrt( gravity * hx_plus ), u_right - u_ic, u_ic - u_left, u_right2 - u_right );
                    u_w_plus( i, j, k, 0 ) *= u_right - u_ic;

                    // DEBUG: Print hx_w_minus, hx_w_plus, u_w_minus, u_w_plus, i, j, k
                    // if ( DEBUG ) std::cout << std::left << std::setw( 10 ) << "hx_w_minus: " << std::setw( 6 ) << hx_w_minus( i, j, k, 0 ( i, j, k, 0 ) << "\thx_w_plus: " << std::setw( 6 ) << hx_w_plus( i, j, k, 0 ) <<\
                    "\tu_w_minus: " << std::setw( 6 ) << u_w_minus( i, j, k, 0 ) << "\tu_w_plus: " << std::setw( 6 ) << u_w_plus( i, j, k, 0 ) << \
                    "\ti: " << i << "\tj: " << j << "\tk: " << k << "\n";

                    // Y Direction
                    hy_w_minus( i, j, k, 0 ) = wCorrector( dt, dy, fabs( vy_minus / hy_minus ) + sqrt( gravity * hy_minus ), h_ic - h_bot, h_bot - h_bot2, h_top - h_ic );
                    hy_w_minus( i, j, k, 0 ) *= h_ic - h_bot;

                    hy_w_plus( i, j, k, 0 ) = wCorrector( dt, dy, fabs( vy_plus / hy_plus ) + sqrt( gravity * hy_plus ), h_top - h_ic, h_ic - h_bot, h_top2 - h_top );
                    hy_w_plus( i, j, k, 0 ) *= h_top - h_ic;

                    u_w_minus( i, j, k, 1 ) = wCorrector( dt, dy, fabs( vy_minus / hy_minus ) + sqrt( gravity * hy_minus ), v_ic - v_bot, v_bot - v_bot2, v_top - v_ic );
                    u_w_minus( i, j, k, 1 ) *= v_ic - v_bot;

                    u_w_plus( i, j, k, 1 ) = wCorrector( dt, dy, fabs( vy_plus / hy_plus ) + sqrt( gravity * hy_plus ), v_top - v_ic, v_ic - v_bot, v_top2 - v_top );
                    u_w_plus( i, j, k, 1 ) *= v_top - v_ic;

                    // DEBUG: Print hy_w_minus, hy_w_plus, u_w_minus, u_w_plus, i, j, k
                    // if ( DEBUG ) std::cout << std::left << std::setw( 10 ) << "hy_w_minus: " << std::setw( 6 ) << hy_w_minus( i, j, k, 0 ( i, j, k, 0 ) << "\thy_w_plus: " << std::setw( 6 ) << hy_w_plus( i, j, k, 0 ) <<\
                    "\tu_w_minus: " << std::setw( 6 ) << u_w_minus( i, j, k, 1 ) << "\tu_w_plus: " << std::setw( 6 ) << u_w_plus( i, j, k, 1 ) << \
                    "\ti: " << i << "\tj: " << j << "\tk: " << k << "\n";

                    // Full Step Update
                    h_new( i, j, k, 0 ) = uFullStep( dt, dx, h_ic, hx_flux_plus( i, j, k, 0 ), hx_flux_minus( i, j, k, 0 ), hy_flux_plus( i, j, k, 0 ), hy_flux_minus( i, j, k, 0 ) ) - hx_w_minus( i, j, k, 0 ) + hx_w_plus( i, j, k, 0 ) - hy_w_minus( i, j, k, 0 ) + hy_w_plus( i, j, k, 0 );
                    u_new( i, j, k, 0 ) = uFullStep( dt, dx, u_ic, ux_flux_plus( i, j, k, 0 ), ux_flux_minus( i, j, k, 0 ), uy_flux_plus( i, j, k, 0 ), uy_flux_minus( i, j, k, 0 ) ) - u_w_minus( i, j, k, 0 ) + u_w_plus( i, j, k, 0 );
                    u_new( i, j, k, 1 ) = uFullStep( dt, dy, v_ic, ux_flux_plus( i, j, k, 1 ), ux_flux_minus( i, j, k, 1 ), uy_flux_plus( i, j, k, 1 ), uy_flux_minus( i, j, k, 1 ) ) - u_w_minus( i, j, k, 1 ) + u_w_plus( i, j, k, 1 );

                    // DEBUG: Print h_new, u_new, v_new, i, j, k
                    // if ( DEBUG ) std::cout << std::left << std::setw( 10 ) << "h_new: " << std::setw( 6 ) << h_new( i, j, k, 0 ) << \
                    "\tu_new: " << std::setw( 6 ) << u_new( i, j, k, 0 ) << "\tv_new: " << std::setw( 6 ) << u_new( i, j, k, 1 ) << "\ti: " << i << "\tj: " << j << "\tk: " << k << "\n";
                } );
        }

    } // namespace TimeIntegrator

} // namespace ExaCLAMR

#endif

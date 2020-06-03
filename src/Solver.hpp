/**
 * @file
 * @author Patrick Bridges
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 *
 * @version 0.1.0
 * 
 * @section LICENSE
 * 
 * @section DESCRIPTION
 * 
 */
 
#ifndef EXACLAMR_SOLVER_HPP
#define EXACLAMR_SOLVER_HPP

#include <Mesh.hpp>
#include <ProblemManager.hpp>
#include <TimeIntegration.hpp>

#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>
#include <memory>

namespace ExaCLAMR
{


class SolverBase
{
    public:
        virtual ~SolverBase() = default;
        virtual void solve( const double t_final, const int write_freq ) = 0;
};

template<class MemorySpace, class ExecutionSpace>
class Solver : public SolverBase
{
    public:

        /**
         * Constructor
         * 
         * @param
         */
        template<class InitFunc>
        Solver( MPI_Comm comm, 
                const InitFunc& create_functor,
                const std::array<double, 6>& global_bounding_box, 
                const std::array<int, 3>& global_num_cell, 
                const std::array<bool, 3>& periodic,
                const Cajita::Partitioner& partitioner,
                const int halo_size, 
                const double dt, 
                const double gravity )
        : _halo_size ( halo_size ), _dt ( dt ), _gravity ( gravity )
        {
            MPI_Comm_rank( comm, &_rank );

            if ( _rank == 0 ) printf( "Created Solver\n" );

            _mesh = std::make_shared<Mesh<MemorySpace>> ( global_bounding_box, 
                                            global_num_cell, 
                                            periodic, 
                                            partitioner, 
                                            halo_size, 
                                            comm );
                
            _pm = std::make_shared<ProblemManager<MemorySpace, double>>( _mesh, create_functor, ExecutionSpace() );

            MPI_Barrier( MPI_COMM_WORLD );
        };

        void output( const int rank, const int tstep ) {
            int a, b;
            if ( tstep % 2 == 0 ) {
                a = 0;
                b = 1;
            }
            else {
                a = 1;
                b = 0;
            }

            auto owned_cells = _pm->mesh()->localGrid()->indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );

            auto uNew = _pm->get( Location::Cell(), Field::Velocity(), b );
            auto hNew = _pm->get( Location::Cell(), Field::Height(), b );

            if ( _pm->mesh()->rank() == rank ) {
                for ( int i = owned_cells.min( 0 ); i < owned_cells.max( 0 ); i++ ) {
                    for ( int j = owned_cells.min( 1 ); j < owned_cells.max( 1 ); j++ ) {
                        for ( int k = owned_cells.min( 2 ); k < owned_cells.max( 2 ); k++ ) {
                            printf( "%.4f\t", hNew( i, j, k, 0 ) );
                        }
                    }
                    printf("\n");
                }
            }
        }

        void solve( const double t_final, const int write_freq ) override {
            if ( _rank == 0 ) printf( "Solving!\n" );

            int nt = round( t_final / _dt );
            double current_time = 0.0;

            if (_rank == 0 ) {
                printf( "Current Time: %.4f\n", current_time );
            }

            for (int t = 1; t <= nt; t++) {

                TimeIntegrator::step( *_pm, ExecutionSpace(), MemorySpace(), _dt, _gravity, t );
                current_time += _dt;

                if ( 0 == t % write_freq ) {
                    if ( 0 == _rank ) printf( "Current Time: %.4f\n", current_time );

                    MPI_Barrier( MPI_COMM_WORLD );

                    output( 0, t );
                }
            }
        };

    private:

        double _dt;
        double _gravity;
        int _halo_size;
        int _rank;
        std::shared_ptr<Mesh<MemorySpace>> _mesh;
        std::shared_ptr<ProblemManager<MemorySpace, double>> _pm;
};

template<class InitFunc>
std::shared_ptr<SolverBase> createSolver( const std::string& device,
                                            MPI_Comm comm, 
                                            const InitFunc& create_functor,
                                            const std::array<double, 6>& global_bounding_box, 
                                            const std::array<int, 3>& global_num_cell,
                                            const std::array<bool, 3>& periodic,
                                            const Cajita::Partitioner& partitioner, 
                                            const int halo_size, 
                                            const double dt, 
                                            const double gravity ) 
{
    if ( 0 == device.compare( "serial" ) ){
        #ifdef KOKKOS_ENABLE_SERIAL
            return std::make_shared<ExaCLAMR::Solver<Kokkos::HostSpace, Kokkos::Serial>>(
                comm, 
                create_functor,
                global_bounding_box, 
                global_num_cell, 
                periodic,
                partitioner,
                halo_size, 
                dt, 
                gravity );
        #else
            throw std::runtime_Error( "Serial Backend Not Enabled" );
        #endif
    }
    else if ( 0 == device.compare( "openmp" ) ) {
        #ifdef KOKKOS_ENABLE_OPENMP
            return std::make_shared<ExaCLAMR::Solver<Kokkos::HostSpace, Kokkos::OpenMP>>(
                comm, 
                create_functor,
                global_bounding_box, 
                global_num_cell, 
                periodic,
                partitioner,
                halo_size, 
                dt, 
                gravity );
        #else
            throw std::runtime_Error( "OpenMP Backend Not Enabled" );
        #endif
    }
    else {
        throw std::runtime_error( "Invalid Backend" );
        return nullptr;
    }
};

}
#endif
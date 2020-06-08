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
        virtual void solve( const int write_freq ) = 0;
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
                const double t_steps, 
                const double gravity )
        : _halo_size ( halo_size ), _tsteps ( t_steps ), _gravity ( gravity )
        {
            MPI_Comm_rank( comm, &_rank );

            if ( _rank == 0 ) printf( "Created Solver\n" );

            _mesh = std::make_shared<Mesh<MemorySpace, ExecutionSpace>> ( global_bounding_box, 
                                            global_num_cell, 
                                            periodic, 
                                            partitioner, 
                                            halo_size, 
                                            comm,
                                            ExecutionSpace()
                                            );
                
            _pm = std::make_shared<ProblemManager<MemorySpace, ExecutionSpace, double>>( _mesh, create_functor, ExecutionSpace() );

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
            auto domain = _pm->mesh()->domainSpace();

            auto uNew = _pm->get( Location::Cell(), Field::Velocity(), b );
            auto hNew = _pm->get( Location::Cell(), Field::Height(), b );

            double summedHeight = 0;
            if ( _pm->mesh()->rank() == rank ) {
                for ( int i = domain.min( 0 ); i < domain.max( 0 ); i++ ) {
                    for ( int j = domain.min( 1 ); j < domain.max( 1 ); j++ ) {
                        for ( int k = domain.min( 2 ); k < domain.max( 2 ); k++ ) {
                            printf( "%-8.4f\t", hNew( i, j, k, 0 ) );
                            summedHeight += hNew( i, j, k, 0 );
                        }
                    }
                    printf("\n");
                }

                // Proxy Mass Conservation
                printf( "Summed Height: %.4f\n", summedHeight );
            }
        };

        void solve( const int write_freq ) override {
            if ( _rank == 0 ) printf( "Solving!\n" );

            int nt = _tsteps;
            double current_time = 0.0;
            int a, b;

            if (_rank == 0 ) {
                printf( "Current Time: %.4f\n", current_time );
                output( 0 , 0 );
            }

            for (int t = 1; t <= nt; t++) {
                if ( t % 2 == 0 ) {
                    a = 0;
                    b = 1;
                }
                else {
                    a = 1;
                    b = 0;
                }

                double mindt;
                double dt = TimeIntegrator::setTimeStep( *_pm, ExecutionSpace(), MemorySpace(), _gravity, 0.95, 0, 1 );

                MPI_Allreduce( &dt, &mindt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );

                TimeIntegrator::step( *_pm, ExecutionSpace(), MemorySpace(), mindt, _gravity, a, b );
                current_time += mindt;

                if ( 0 == t % write_freq ) {
                    if ( 0 == _rank ) printf( "Current Time: %.4f\n", current_time );

                    MPI_Barrier( MPI_COMM_WORLD );

                    output( 0, t );
                }
            }
        };

    private:

        double _tsteps;
        double _gravity;
        int _halo_size;
        int _rank;
        std::shared_ptr<Mesh<MemorySpace, ExecutionSpace>> _mesh;
        std::shared_ptr<ProblemManager<MemorySpace, ExecutionSpace, double>> _pm;
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
                                            const double t_steps, 
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
                t_steps, 
                gravity );
        #else
            throw std::runtime_error( "Serial Backend Not Enabled" );
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
                t_steps, 
                gravity );
        #else
            throw std::runtime_error( "OpenMP Backend Not Enabled" );
        #endif
    }
    else {
        throw std::runtime_error( "Invalid Backend" );
        return nullptr;
    }
};

}
#endif

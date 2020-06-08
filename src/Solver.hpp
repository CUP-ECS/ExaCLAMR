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

#ifdef HAVE_SILO
    #include <silo.h>
#endif

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
                const double rFill,
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
                
            _pm = std::make_shared<ProblemManager<MemorySpace, ExecutionSpace, double>>( rFill, _mesh, create_functor, ExecutionSpace() );

            MPI_Barrier( MPI_COMM_WORLD );
        };

        void writeFile( DBfile *dbfile, char *name, int cycle, double time, double dtime, int b ) {
            int            i, dims[2], zdims[2], zones[2], ndims, meshid;
            float          x[5], y[5], d[25], *coords[2], *vars[2];
            float          u[5], v[5];
            char          *coordnames[2], *varnames[2];
            DBoptlist     *optlist;

            optlist = DBMakeOptlist(10);
            DBAddOption(optlist, DBOPT_CYCLE, &cycle);
            DBAddOption(optlist, DBOPT_TIME, &time);
            DBAddOption(optlist, DBOPT_DTIME, &dtime);

            ndims = 2;
            dims[0] = 5;
            dims[1] = 5;

            zones[0] = 4;
            zones[1] = 4;

            coords[0] = x;
            coords[1] = y;

            coordnames[0] = strdup( "x" );
            coordnames[1] = strdup( "y" );

            for (i = 0; i < dims[0]; i++)
                x[i] = (float)i;
            for (i = 0; i < dims[1]; i++)
                y[i] = (float)i;

            meshid = DBPutQuadmesh(dbfile, name, (DBCAS_t) coordnames,
                        coords, dims, ndims, DB_FLOAT, DB_COLLINEAR, optlist);


            auto owned_cells = _pm->mesh()->localGrid()->indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );
            auto domain = _pm->mesh()->domainSpace();

            auto uNew = _pm->get( Location::Cell(), Field::Velocity(), b );
            auto hNew = _pm->get( Location::Cell(), Field::Height(), b );

            varnames[0] = strdup( "d" );
            vars[0] = d;
            zdims[0] = dims[0] - 1;
            zdims[1] = dims[1] - 1;

            for (i = 0; i < zdims[0] * zdims[1]; i++)
                d[i] = (float)i  * .2;

            DBPutQuadvar1(dbfile, "d", name, d, zdims, ndims,
                                NULL, 0, DB_FLOAT, DB_ZONECENT, optlist);

            /*
            for (i = 0; i < dims[0] * dims[1]; i++) {
                u[i] = (float)i *.1;
                v[i] = (float)i *.1;
            }

            DBPutQuadvar1(dbfile, "ucomp", name, u, dims, ndims,
                                NULL, 0, DB_FLOAT, DB_NODECENT, optlist);
            DBPutQuadvar1(dbfile, "vcomp", name, v, dims, ndims,
                                NULL, 0, DB_FLOAT, DB_NODECENT, optlist);

            vars[0] = u;
            vars[1] = v;
            varnames[0] = "u";
            varnames[1] = "v";

            DBPutQuadvar(dbfile, "velocity", name, 2, (DBCAS_t) varnames,
                vars, dims, ndims, NULL, 0, DB_FLOAT, DB_NODECENT, optlist);
            */

            DBFreeOptlist(optlist);
        }

        void output( const int rank, const int tstep, const double current_time, const double dt ) {
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

            #ifdef HAVE_SILO
            DBfile *silo_file;
            int driver = DB_PDB;

            DBShowErrors( DB_ALL, NULL );
            DBForceSingle( 1 );

            char filename[30];

            sprintf( filename, "ExaCLAMROutput%05d.pdb", tstep );

            if ( rank == 0 && tstep == 0 ) {
                printf("Creating file: `%s'\n", filename);
                silo_file = DBCreate(filename, 0, DB_LOCAL, "ExaCLAMR", driver);

                writeFile( silo_file, strdup( "Mesh" ), tstep, current_time, dt, b );

                DBClose( silo_file );
            }
            else if ( rank == 0 ) {
                printf("Creating file: `%s'\n", filename);
                silo_file = DBCreate(filename, 0, DB_LOCAL, "ExaCLAMR", driver);

                writeFile( silo_file, strdup( "Mesh" ), tstep, current_time, dt, b );

                DBClose( silo_file );
            }
            #endif
        };

        void solve( const int write_freq ) override {
            if ( _rank == 0 ) printf( "Solving!\n" );

            int nt = _tsteps;
            double current_time = 0.0;
            int a, b;

            if (_rank == 0 ) {
                printf( "Current Time: %.4f\n", current_time );
                output( 0, 0, 0, 0 );
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

                    output( 0, t, current_time, mindt );
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
                                            const double rFill,
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
                rFill,
                periodic,
                partitioner,
                halo_size, 
                t_steps, 
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
                rFill,
                periodic,
                partitioner,
                halo_size, 
                t_steps, 
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
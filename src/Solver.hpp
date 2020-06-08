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
            int            dims[2], zdims[2], zones[2], ndims, meshid;
            double        *coords[2], *vars[2];
            char          *coordnames[2], *varnames[2];
            DBoptlist     *optlist;

            optlist = DBMakeOptlist(10);
            DBAddOption(optlist, DBOPT_CYCLE, &cycle);
            DBAddOption(optlist, DBOPT_TIME, &time);
            DBAddOption(optlist, DBOPT_DTIME, &time);

            coordnames[0] = strdup( "x" );
            coordnames[1] = strdup( "y" );

            auto domain = _pm->mesh()->domainSpace();
            int nx = domain.extent( 0 );
            int ny = domain.extent( 1 );

            ndims = 2;
            dims[0] = nx + 1;
            dims[1] = ny + 1;

            double x[dims[0]];
            double y[dims[1]];
            double height[nx * ny];
            double u[nx * ny];
            double v[nx * ny];

            double dx = _pm->mesh()->localGrid()->globalGrid().globalMesh().cellSize( 0 );
            double dy = _pm->mesh()->localGrid()->globalGrid().globalMesh().cellSize( 1 );

            coords[0] = x;
            coords[1] = y;

            for (int i = 0; i < dims[0]; i++) {
                x[i] = ( double ) i * dx;
            }
            for (int j = 0; j < dims[1]; j++) {
                y[j] = ( double ) j * dy;
            }

            meshid = DBPutQuadmesh(dbfile, name, (DBCAS_t) coordnames,
                        coords, dims, ndims, DB_DOUBLE, DB_COLLINEAR, optlist);


            auto owned_cells = _pm->mesh()->localGrid()->indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );

            auto uNew = _pm->get( Location::Cell(), Field::Velocity(), b );
            auto hNew = _pm->get( Location::Cell(), Field::Height(), b );


            zones[0] = dims[0] - 1;
            zones[1] = dims[1] - 1;

            varnames[0] = strdup( "Height" );
            vars[0] = height;

            zdims[0] = dims[0] - 1;
            zdims[1] = dims[1] - 1;

            for ( int i = domain.min( 0 ); i < domain.max( 0 ); i++ ) {
                for ( int j = domain.min( 1 ); j < domain.max( 1 ); j++ ) {
                    for ( int k = domain.min( 2 ); k < domain.max( 2 ); k++ ) {
                        int iown = i - domain.min( 0 );
                        int jown = j - domain.min( 1 );
                        int kown = k - domain.min( 2 );
                        int inx = iown + domain.extent( 0 ) * ( jown + domain.extent( 1 ) * kown );
                        height[inx] = hNew( i, j, k, 0 );
                        u[inx] = uNew( i, j, k, 0 );
                        v[inx] = uNew( i, j, k, 1 );
                    }
                }
            }

            DBPutQuadvar1( dbfile, "height", name, height, zdims, ndims,
                                NULL, 0, DB_DOUBLE, DB_ZONECENT, optlist );
            
            DBPutQuadvar1( dbfile, "ucomp", name, u, zdims, ndims,
                        NULL, 0, DB_DOUBLE, DB_ZONECENT, optlist );

            DBPutQuadvar1( dbfile, "vcomp", name, v, zdims, ndims,
                        NULL, 0, DB_DOUBLE, DB_ZONECENT, optlist );

            vars[0] = u;
            vars[1] = v;
            varnames[0] = strdup( "u" );
            varnames[1] = strdup( "v" );


            DBPutQuadvar( dbfile, "velocity", name, 2, (DBCAS_t) varnames,
                vars, zdims, ndims, NULL, 0, DB_DOUBLE, DB_ZONECENT, optlist );

            DBFreeOptlist( optlist );
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
                            // printf( "%-8.4f\t", hNew( i, j, k, 0 ) );
                            summedHeight += hNew( i, j, k, 0 );
                        }
                    }
                    // printf("\n");
                }

                // Proxy Mass Conservation
                printf( "Summed Height: %.4f\n", summedHeight );
            }

            #ifdef HAVE_SILO
            DBfile *silo_file;
            int driver = DB_PDB;
            char filename[30];

            sprintf( filename, "data/ExaCLAMROutput%05d.pdb", tstep );

            DBShowErrors( DB_ALL, NULL );
            DBForceSingle( 1 );

            if ( rank == 0 ) {
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
/**
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * Silo Writer class to write results to a silo file using PMPIO
 */

#ifndef EXACLAMR_SILOWRITER_HPP
#define EXACLAMR_SILOWRITER_HPP

#ifndef DEBUG
    #define DEBUG 0 
#endif

// Include Statements
#include <ExaCLAMR.hpp>

#include <Cajita.hpp>

#ifdef HAVE_SILO
    #include <silo.h>
    #include <pmpio.h>
#endif


namespace ExaCLAMR {

/**
 * @struct SiloTraits
 * @brief Parent SiloTraits Struct to template and allow writing floats or doubles to silo file
 **/
template <typename SiloType>
struct SiloTraits;

/**
 * @struct SiloTraits<float>
 * @brief Child SiloTraits Struct to write floats to silo file
 **/
template <>
struct SiloTraits<float> {
    static DBdatatype type() { return DB_FLOAT; }
};

/**
 * @struct SiloTraits<double>
 * @brief Child SiloTraits Struct to write doubles to silo file
 **/
template <>
struct SiloTraits<double> {
    static DBdatatype type() { return DB_DOUBLE; }
};


/**
 * The SiloWriter Class
 * @class SiloWriter
 * @brief SiloWriter class to write results to Silo file using PMPIO
 **/
template <class MeshType, class MemorySpace, class ExecutionSpace>
class SiloWriter;

template <class state_t, class MemorySpace, class ExecutionSpace>
class SiloWriter<ExaCLAMR::AMRMesh<state_t>, MemorySpace, ExecutionSpace> {
    public:
        SiloWriter() {
            std::cout << "AMR Silo\n";
        }
};

template <class state_t, class MemorySpace, class ExecutionSpace>
class SiloWriter<ExaCLAMR::RegularMesh<state_t>, MemorySpace, ExecutionSpace> {
    public:
        /**
         * Constructor
         * Create new SiloWriter
         * 
         * @param pm Problem manager object
         */
        template<class ProblemManagerType>
        SiloWriter( ProblemManagerType& pm ) 
        : _pm ( pm ) {};



        /**
         * Write File
         * @param dbile File handler to dbfile
         * @param name File name
         * @param time_step Current time step
         * @param time Current tim
         * @param dt Time Step (dt)
         **/
        void writeFile( DBfile *dbfile, char *name, int time_step, state_t time, state_t dt ) {
            // Initialize Variables
            int            dims[2], zdims[2], nx, ny, ndims;
            state_t       *coords[2], *vars[2], dx, dy;
            char          *coordnames[2], *varnames[2];
            DBoptlist     *optlist;

            // Define device_type for Later Use
            using device_type = typename Kokkos::Device<ExecutionSpace, MemorySpace>;

            // Create Local Grid
            auto local_grid = _pm->mesh()->localGrid();

            // Get Local Mesh and Owned Cells for Domain Calculation
            auto local_mesh = Cajita::createLocalMesh<device_type>( *local_grid );

            // DEBUG: Trace Writing File
            if( DEBUG ) std::cout << "Writing File\n";

            // Set DB Options: Time Step, Time Stamp and Delta Time
            optlist = DBMakeOptlist(10);
            DBAddOption(optlist, DBOPT_CYCLE, &time_step);
            DBAddOption(optlist, DBOPT_TIME, &time);
            DBAddOption(optlist, DBOPT_DTIME, &dt);

            // Get Domain Space
            auto domain = _pm->mesh()->domainSpace();

            // Get Number of Cells and Cell Size in 2-Dimensions ( X, Y )
            nx = domain.extent( 0 );
            ny = domain.extent( 1 );
            dx = local_grid->globalGrid().globalMesh().cellSize( 0 );
            dy = local_grid->globalGrid().globalMesh().cellSize( 1 );

            // 2-D Cell-Centered Regular Mesh
            ndims = 2;
            // Account for Edge Node
            dims[0] = nx + 1;
            dims[1] = ny + 1;

            // Correct Number of Cells/Zones
            zdims[0] = dims[0] - 1;         // Equivalent to nx
            zdims[1] = dims[1] - 1;         // Equivalent to ny

            // Coordinate Names: Cartesian X, Y Coordinate System
            coordnames[0] = strdup( "x" );
            coordnames[1] = strdup( "y" );

            // Initialize Coordinate and State Arrays for Writing
            state_t x[dims[0]], y[dims[1]];
            state_t height[nx * ny], u[nx * ny], v[nx * ny];

            // Point Coords to X and Y Coordinates
            coords[0] = x;
            coords[1] = y;

            // Set X and Y Coordinates of Nodes
            for (int i = domain.min( 0 ); i <= domain.max( 0 ); i++) {
                int iown = i - domain.min( 0 );
                int coords[3] = { i, 0, 0 };
                state_t x_coords[3];
                local_mesh.coordinates( Cajita::Cell(), coords, x_coords );
                x[iown] = x_coords[0] - 0.5 * dx;
            }

            for (int j = domain.min( 1 ); j <= domain.max( 1 ); j++) {
                int jown = j - domain.min( 1 );
                int coords[3] = { 0, j, 0 };
                state_t x_coords[3];
                local_mesh.coordinates( Cajita::Cell(), coords, x_coords );
                y[jown] = x_coords[1] - 0.5 * dy;
            }

            DBPutQuadmesh(dbfile, name, (DBCAS_t) coordnames,
                        coords, dims, ndims, SiloTraits<state_t>::type(), DB_COLLINEAR, optlist);

            // Get State Views
            auto uNew = _pm->get( Location::Cell(), Field::Momentum(), NEWFIELD( time_step ) );
            auto hNew = _pm->get( Location::Cell(), Field::Height(), NEWFIELD( time_step ) );

            // Loop Over Domain ( i, j, k )
            for ( int i = domain.min( 0 ); i < domain.max( 0 ); i++ ) {
                for ( int j = domain.min( 1 ); j < domain.max( 1 ); j++ ) {
                    for ( int k = domain.min( 2 ); k < domain.max( 2 ); k++ ) {
                        // Adjust Indices to Start at ( 0, 0, 0 ) - Account for Offset from Boundary Cells
                        int iown = i - domain.min( 0 );
                        int jown = j - domain.min( 1 );
                        int kown = k - domain.min( 2 );
                        // Calculate 1-Dimensional Index from iown, jown, and kown
                        int inx = iown + domain.extent( 0 ) * ( jown + domain.extent( 1 ) * kown );

                        // Set State Values to be Written
                        height[inx] = hNew( i, j, k, 0 );
                        u[inx] = uNew( i, j, k, 0 );
                        v[inx] = uNew( i, j, k, 1 );
                    }
                }
            }

            // Write Scalar Variables
            // Height
            DBPutQuadvar1( dbfile, "height", name, height, zdims, ndims,
                                NULL, 0, SiloTraits<state_t>::type(), DB_ZONECENT, optlist );
            
            // Vx
            DBPutQuadvar1( dbfile, "ucomp", name, u, zdims, ndims,
                        NULL, 0, SiloTraits<state_t>::type(), DB_ZONECENT, optlist );

            // Vy
            DBPutQuadvar1( dbfile, "vcomp", name, v, zdims, ndims,
                        NULL, 0, SiloTraits<state_t>::type(), DB_ZONECENT, optlist );

            // Setup and Write Momentum Variable
            vars[0] = u;
            vars[1] = v;
            varnames[0] = strdup( "u" );
            varnames[1] = strdup( "v" );

            // Momentum
            DBPutQuadvar( dbfile, "momentum", name, 2, (DBCAS_t) varnames,
                vars, zdims, ndims, NULL, 0, SiloTraits<state_t>::type(), DB_ZONECENT, optlist );

            // Free Option List
            DBFreeOptlist( optlist );
        };

        /**
         * Create New Silo File for Current Time Step and Owning Group
         * @param filename Name of file
         * @param nsname Name of directory inside of the file
         * @param user_data File Driver/Type (PDB, HDF5)
         **/
        static void* createSiloFile( const char* filename, const char* nsname, void* user_data ) {
            if ( DEBUG ) std::cout << "Creating file: " << filename << "\n";

            int driver = *( ( int* ) user_data );
            DBfile* silo_file = DBCreate( filename, DB_CLOBBER, DB_LOCAL, "ExaCLAMRRaw", driver );

            if ( silo_file ) {
                DBMkDir( silo_file, nsname );
                DBSetDir( silo_file, nsname );
            }

            return ( void * ) silo_file;
        };

        /**
         * Open Silo File
         * @param filename Name of file
         * @param nsname Name of directory inside of file
         * @param ioMode Read/Write/Append Mode
         * @param user_data File Driver/Type (PDB, HDF5)
         **/
        static void* openSiloFile( const char* filename, const char* nsname, PMPIO_iomode_t ioMode, void* user_data ) {
            DBfile* silo_file = DBOpen( filename, DB_UNKNOWN, ioMode == PMPIO_WRITE ? DB_APPEND : DB_READ );

            if ( silo_file ) {
                if ( ioMode == PMPIO_WRITE ) {
                    DBMkDir( silo_file, nsname );
                }
                DBSetDir( silo_file, nsname );
            }

            return ( void * ) silo_file;
        };

        /**
         * Close Silo File
         * @param file File pointer
         * @param user_data File Driver/Type (PDB, HDF5)
         **/
        static void closeSiloFile( void* file, void* user_data ) {
            DBfile* silo_file = ( DBfile * ) file;
            if ( silo_file ) DBClose( silo_file );
        };

        /**
         * Write Multi Object Silo File the References Child Files in order to have entire set of data for the time step within a Single File
         * Combines several Silo Files into a Single Silo File
         * 
         * @param silo_file Pointer to the Silo File
         * @param baton Baton object from PMPIO
         * @param size Number of Ranks
         * @param time_step Current time step
         * @param file_ext File extension (PDB, HDF5)
         **/
        void writeMultiObjects( DBfile* silo_file, PMPIO_baton_t* baton, int size, int time_step, const char* file_ext ) {
            char** mesh_block_names = ( char ** ) malloc( size * sizeof( char * ) );
            char** h_block_names = ( char ** ) malloc( size * sizeof( char * ) );
            char** u_block_names = ( char ** ) malloc( size * sizeof( char * ) );
            char** v_block_names = ( char ** ) malloc( size * sizeof( char * ) );
            char** mom_block_names = ( char ** ) malloc( size * sizeof( char * ) );

            int* block_types = ( int * ) malloc( size * sizeof( int ) );
            int* var_types = ( int * ) malloc( size * sizeof( int ) );

            DBSetDir( silo_file, "/" );

            for ( int i = 0; i < size; i++ ) {
                int group_rank = PMPIO_GroupRank( baton, i );
                mesh_block_names[i] = ( char * ) malloc( 1024 );
                h_block_names[i] = ( char * ) malloc( 1024 );
                u_block_names[i] = ( char * ) malloc( 1024 );
                v_block_names[i] = ( char * ) malloc( 1024 );
                mom_block_names[i] = ( char * ) malloc( 1024 );

                sprintf( mesh_block_names[i], "raw/ExaCLAMROutput%05d%05d.pdb:/domain_%05d/Mesh", group_rank, time_step, i );
                sprintf( h_block_names[i], "raw/ExaCLAMROutput%05d%05d.pdb:/domain_%05d/height", group_rank, time_step, i );
                sprintf( u_block_names[i], "raw/ExaCLAMROutput%05d%05d.pdb:/domain_%05d/ucomp", group_rank, time_step, i );
                sprintf( v_block_names[i], "raw/ExaCLAMROutput%05d%05d.pdb:/domain_%05d/vcomp", group_rank, time_step, i );
                sprintf( mom_block_names[i], "raw/ExaCLAMROutput%05d%05d.pdb:/domain_%05d/momentum", group_rank, time_step, i );

                block_types[i] = DB_QUADMESH;
                var_types[i] = DB_QUADVAR;
            }

            DBPutMultimesh( silo_file, "multi_mesh", size, mesh_block_names, block_types, 0 );
            DBPutMultivar( silo_file, "multi_height", size, h_block_names, var_types, 0 );
            DBPutMultivar( silo_file, "multi_ucomp", size, u_block_names, var_types, 0 );
            DBPutMultivar( silo_file, "multi_vcomp", size, v_block_names, var_types, 0 );
            DBPutMultivar( silo_file, "multi_momentum", size, mom_block_names, var_types, 0 );

            for ( int i = 0; i < size; i++ ) {
                free( mesh_block_names[i] );
                free( h_block_names[i] );
                free( u_block_names[i] );
                free( v_block_names[i] );
                free( mom_block_names[i] );
            }

            free( mesh_block_names );
            free( h_block_names );
            free( u_block_names );
            free( v_block_names );
            free( mom_block_names );
            free( block_types );
            free( var_types );
        }

        // Function to Create New DB File for Current Time Step
        /**
         * Createe New DB File for Current Time Step
         * @param name Name of directory in silo file
         * @param time_step Current time step
         * @param time Current time
         * @param dt Time step (dt)
         **/
        void siloWrite( char *name, int time_step, state_t time, state_t dt ) {
            // Initalize Variables
            DBfile* silo_file;
            DBfile* master_file;
            int size;
            int driver = DB_PDB;
            // TODO: Make the Number of Groups a Constant or a Runtime Parameter ( Between 8 and 64 )
            int numGroups = 2;
            char masterfilename[256], filename[256], nsname[256];
            PMPIO_baton_t* baton;

            MPI_Comm_size(MPI_COMM_WORLD, &size);
            MPI_Bcast( &numGroups, 1, MPI_INT, 0, MPI_COMM_WORLD );
            MPI_Bcast( &driver, 1, MPI_INT, 0, MPI_COMM_WORLD );

            baton = PMPIO_Init( numGroups, PMPIO_WRITE, MPI_COMM_WORLD, 1, createSiloFile, openSiloFile, closeSiloFile, &driver );

            // Set Filename to Reflect TimeStep
            sprintf( masterfilename, "data/ExaCLAMR%05d.pdb", time_step );
            sprintf( filename, "data/raw/ExaCLAMROutput%05d%05d.pdb", PMPIO_GroupRank( baton, _pm->mesh()->rank() ), time_step );
            sprintf( nsname, "domain_%05d", _pm->mesh()->rank() );

            // Show Errors and Force FLoating Point
            DBShowErrors( DB_ALL, NULL );

            silo_file = ( DBfile * ) PMPIO_WaitForBaton( baton, filename, nsname );

            writeFile( silo_file, name, time_step, time, dt );

            if ( _pm->mesh()->rank() == 0 ) {
                master_file = DBCreate( masterfilename, DB_CLOBBER, DB_LOCAL, "ExaCLAMR", driver );
                writeMultiObjects( master_file, baton, size, time_step, "pdb" );
                DBClose( master_file );
            }

            PMPIO_HandOffBaton( baton, silo_file );

            PMPIO_Finish( baton );
            }

    private:
        std::shared_ptr<ProblemManager<ExaCLAMR::RegularMesh<state_t>, MemorySpace, ExecutionSpace>> _pm;      /**< Problem Manager Shared Pointer */

};

};

#endif

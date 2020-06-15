/**
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * 
 */

#ifndef EXACLAMR_SILOWRITER_HPP
#define EXACLAMR_SILOWRITER_HPP

#ifndef DEBUG
    #define DEBUG 0 
#endif

#ifdef HAVE_SILO
    #include <silo.h>
#endif


namespace ExaCLAMR {

template <typename SiloType>
struct SiloTraits;

template <>
struct SiloTraits<float> {
    static DBdatatype type() { return DB_FLOAT; }
};

template <>
struct SiloTraits<double> {
    static DBdatatype type() { return DB_DOUBLE; }
};

template<class MemorySpace, class ExecutionSpace, typename state_t>
class SiloWriter
{

    public:
        /**
         * Constructor
         * 
         * @param
         */
        template<class ProblemManagerType>
        SiloWriter( ProblemManagerType& pm ) 
        : _pm ( pm ) { }

        // Toggle Between Current and New State Vectors
        #define NEWFIELD( time_step ) ( ( time_step + 1 ) % 2 )
        #define CURRENTFIELD( time_step ) ( ( time_step ) % 2 )

        // Function to Write File in Serial
        // TODO: PMPIO Write File in Parallel
        void writeFile( DBfile *dbfile, char *name, int time_step, state_t time, state_t dt ) {
            // Initialize Variables
            int            dims[2], zdims[2], zones[2], nx, ny, ndims, meshid;
            state_t       *coords[2], *vars[2], dx, dy;
            char          *coordnames[2], *varnames[2];
            DBoptlist     *optlist;

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
            dx = _pm->mesh()->localGrid()->globalGrid().globalMesh().cellSize( 0 );
            dy = _pm->mesh()->localGrid()->globalGrid().globalMesh().cellSize( 1 );

            // 2-D Cell-Centered Regular Mesh
            ndims = 2;
            // Account for Edge Node
            dims[0] = nx + 1;
            dims[1] = ny + 1;
            // Get Correct Number of Cells
            zones[0] = dims[0] - 1;         // Equivalent to nx
            zones[1] = dims[1] - 1;         // Equivalent to ny

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
            for (int i = 0; i < dims[0]; i++) x[i] = ( state_t ) i * dx;
            for (int j = 0; j < dims[1]; j++) y[j] = ( state_t ) j * dy;

            meshid = DBPutQuadmesh(dbfile, name, (DBCAS_t) coordnames,
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


        // Function to Create New DB File for Current Time Step
        void siloWrite( char *name, int time_step, state_t time, state_t dt ) {
            // Initalize Variables
            DBfile *silo_file;
            int driver = DB_PDB;
            char filename[30];

            // Set Filename to Reflect TimeStep
            sprintf( filename, "data/ExaCLAMROutput%05d.pdb", time_step );

            // Show Errors and Force FLoating Point
            DBShowErrors( DB_ALL, NULL );
            DBForceSingle( 1 );

            // Only Rank 0 Creates the File and then Writes to it
            if ( _pm->mesh()->rank() == 0 ) {
                if ( DEBUG ) std::cout << "Creating file: " << filename << "\n";
                silo_file = DBCreate(filename, 0, DB_LOCAL, "ExaCLAMR", driver);
                writeFile( silo_file, strdup( "Mesh" ), time_step, time, dt );
                DBClose( silo_file );
            }
        };

    private:
        std::shared_ptr<ProblemManager<MemorySpace, ExecutionSpace, state_t>> _pm;

};

}

#endif
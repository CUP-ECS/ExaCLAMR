#ifndef EXACLAMR_SILOWRITER_HPP
#define EXACLAMR_SILOWRITER_HPP

#define DEBUG 0

#ifdef HAVE_SILO
    #include <silo.h>
#endif


namespace ExaCLAMR {

template<class MemorySpace, class ExecutionSpace>
class SiloWriter
{

    public:

        template<class ProblemManagerType>
        SiloWriter( ProblemManagerType& pm ) 
        : _pm ( pm ) { }

        void writeFile( DBfile *dbfile, char *name, int cycle, double time, double dtime, int b ) {
            int            dims[2], zdims[2], zones[2], ndims, meshid;
            double        *coords[2], *vars[2];
            char          *coordnames[2], *varnames[2];
            DBoptlist     *optlist;

            if( DEBUG ) std::cout << "Writing File\n";

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

        void siloWrite( char *name, int cycle, double time, double dtime, int b ) {
            #ifdef HAVE_SILO
                DBfile *silo_file;
                int driver = DB_PDB;
                char filename[30];

                sprintf( filename, "data/ExaCLAMROutput%05d.pdb", cycle );

                DBShowErrors( DB_ALL, NULL );
                DBForceSingle( 1 );

                if ( _pm->mesh()->rank() == 0 ) {
                    if ( DEBUG ) std::cout << "Creating file: " << filename << "\n";
                    silo_file = DBCreate(filename, 0, DB_LOCAL, "ExaCLAMR", driver);
                    writeFile( silo_file, strdup( "Mesh" ), cycle, time, dtime, b );
                    DBClose( silo_file );
                }
            #endif
        };

    private:
        std::shared_ptr<ProblemManager<MemorySpace, ExecutionSpace, double>> _pm;

};

}

#endif
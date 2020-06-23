/**
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * Problem manager class that stores the mesh and the state data and performs scatters and gathers
 */

#ifndef EXACLAMR_PROBLEMMANAGER_HPP
#define EXACLAMR_PROBLEMMANAGER_HPP

#ifndef DEBUG
    #define DEBUG 0 
#endif 

// Include Statements
#include <Mesh.hpp>

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <memory>


namespace ExaCLAMR
{

/**
 * @namespace Location
 * @brief Location namespace to track mesh entities
 **/
namespace Location
{
/**
 * @struct Cell
 * @brief Cell Location Type 
 **/
struct Cell {};  

/**
 * @struct Face
 * @brief Face Location Type 
 **/
struct Face {};

/**
 * @struct Node
 * @brief Node Location Type 
 **/
struct Node {};
}


/**
 * @namespace Field
 * @brief Field namespace to track state array entities
 **/
namespace Field
{
/**
 * @struct Momentum
 * @brief Momentum Field
 **/
struct Momentum {};

/**
 * @struct Height
 * @brief Height Field
 **/
struct Height {};

/**
 * @struct HxFluxPlus
 * @brief Positive Height X-Direction Flux Field
 **/
struct HxFluxPlus {};

/**
 * @struct HxFluxMinus
 * @brief Negative Height X-Direction Flux Field
 **/
struct HxFluxMinus {};

/**
 * @struct UxFluxPlus
 * @brief Positive Momentum X-Direction Flux Field
 **/
struct UxFluxPlus {};

/**
 * @struct UxFluxMinus
 * @brief Negative Momentum X-Direction Flux Field
 **/
struct UxFluxMinus {};

/**
 * @struct HyFluxPlus
 * @brief Positive Height Y-Direction Flux Field
 **/
struct HyFluxPlus {};

/**
 * @struct HyFluxMinus
 * @brief Negative Height Y-Direction Flux Field
 **/
struct HyFluxMinus {};

/**
 * @struct UyFluxPlus
 * @brief Positive Momentum Y-Direction Flux Field
 **/
struct UyFluxPlus {};

/**
 * @struct UyFluxMinus
 * @brief Negative Momentum Y-Direction Flux Field
 **/
struct UyFluxMinus {};

/**
 * @struct HxWPlus
 * @brief Positive X-Direction Height Flux Corrector Field
 **/
struct HxWPlus {};

/**
 * @struct HxWMinus
 * @brief Negative X-Direction Height Flux Corrector Field
 **/
struct HxWMinus {};

/**
 * @struct HyWPlus
 * @brief Positive Y-Direction Height Flux Corrector Field
 **/
struct HyWPlus {};

/**
 * @struct HyWMinus
 * @brief Negative Y-Direction Height Flux Corrector Field
 **/
struct HyWMinus {};

/**
 * @struct UWPlus
 * @brief Positive Momentum Flux Corrector Field
 **/
struct UWPlus {};

/**
 * @struct UWMinus
 * @brief Negative Momentum Flux Corrector Field
 **/
struct UWMinus {};
}


/**
 * The ProblemManager Class
 * @class ProblemManager
 * @brief ProblemManager class to store the mesh and state values, and to perform gathers and scatters.
 **/
template<class MemorySpace, class ExecutionSpace, typename state_t>
class ProblemManager
{
    using cell_array = Cajita::Array<state_t, Cajita::Cell, Cajita::UniformMesh<state_t>, MemorySpace>;
    using halo = Cajita::Halo<state_t, MemorySpace>;

    public:
        /**
         * Constructor
         * Creates a new mesh
         * Creates state cell layouts, halo layouts, and Cajita arrays to store state data
         * Initializes state data
         * 
         * @param cl Command line arguments
         * @param partitioner Cajita MPI partitioner
         * @param comm MPI communicator
         * @param create_functor Initialization function
         */
        template <class InitFunc>
        ProblemManager( const ExaCLAMR::ClArgs<state_t>& cl, const Cajita::Partitioner& partitioner, MPI_Comm comm, const InitFunc& create_functor ) {
            // Create Mesh
            _mesh = std::make_shared<Mesh<MemorySpace, ExecutionSpace, state_t>> ( cl, partitioner, comm );

            // Create Vector and Scalar Layouts
            auto cell_vector_layout = Cajita::createArrayLayout( _mesh->localGrid(), 2, Cajita::Cell() );   // 2-Dimensional ( Momentum )
            auto cell_scalar_layout = Cajita::createArrayLayout( _mesh->localGrid(), 1, Cajita::Cell() );

            // Initialize State Arrays
            // A and B Arrays used to Update State Data without Overwriting
            _momentum_a = Cajita::createArray<state_t, MemorySpace>( "momentum", cell_vector_layout );
            _height_a = Cajita::createArray<state_t, MemorySpace>( "height", cell_scalar_layout );

            _momentum_b = Cajita::createArray<state_t, MemorySpace>( "momentum", cell_vector_layout );
            _height_b = Cajita::createArray<state_t, MemorySpace>( "height", cell_scalar_layout );


            // Initialize Flux Arrays
            _hx_flux_plus = Cajita::createArray<state_t, MemorySpace>( "HxFluxPlus", cell_scalar_layout );
            _hx_flux_minus = Cajita::createArray<state_t, MemorySpace>( "HxFluxMinus", cell_scalar_layout );
            _ux_flux_plus = Cajita::createArray<state_t, MemorySpace>( "UxFluxPlus", cell_vector_layout );
            _ux_flux_minus = Cajita::createArray<state_t, MemorySpace>( "UxFluxMinus", cell_vector_layout );

            _hy_flux_plus = Cajita::createArray<state_t, MemorySpace>( "HyFluxPlus", cell_scalar_layout );
            _hy_flux_minus = Cajita::createArray<state_t, MemorySpace>( "HyFluxMinus", cell_scalar_layout );
            _uy_flux_plus = Cajita::createArray<state_t, MemorySpace>( "UyFluxPlus", cell_vector_layout );
            _uy_flux_minus = Cajita::createArray<state_t, MemorySpace>( "UyFluxMinus", cell_vector_layout );

            // Initialize Flux Corrector Arrays
            _hx_w_plus = Cajita::createArray<state_t, MemorySpace>( "HxWPlus", cell_scalar_layout );
            _hx_w_minus = Cajita::createArray<state_t, MemorySpace>( "HxWMinus", cell_scalar_layout );
            _hy_w_plus = Cajita::createArray<state_t, MemorySpace>( "HyWPlus", cell_scalar_layout );
            _hy_w_minus = Cajita::createArray<state_t, MemorySpace>( "HyWMinus", cell_scalar_layout );

            _u_w_plus = Cajita::createArray<state_t, MemorySpace>( "UWPlus", cell_vector_layout );
            _u_w_minus = Cajita::createArray<state_t, MemorySpace>( "UWMinus", cell_vector_layout );

            // Create Halo Pattern
            auto halo_pattern = Cajita::HaloPattern();
            std::vector<std::array<int, 3>> neighbors;

            // Setting up Stencil ( Left, Right, Top, Bottom )
            for ( int i = -1; i < 2; i++ ) {
                for (int j = -1; j < 2; j++) {
                    if ( ( i == 0 || j == 0 ) && !( i == 0 && j == 0 ) ) {
                        neighbors.push_back( { i, j, 0 } );
                    }
                }
            }

            halo_pattern.setNeighbors( neighbors );

            // Initialize Halo Array Layours
            _cell_vector_halo = Cajita::createHalo<state_t, MemorySpace>( *cell_vector_layout, halo_pattern );
            _cell_scalar_halo = Cajita::createHalo<state_t, MemorySpace>( *cell_scalar_layout, halo_pattern );

            // Initialize State Values ( Height, Momentum )
            initialize( create_functor );

        };

        /**
         * Initializes state values in the cells
         * @param create_functor Initialization function
         **/
        template<class InitFunctor>
        void initialize( const InitFunctor& create_functor ) {
            // DEBUG: Trace State Initialization
            if ( _mesh->rank() == 0 && DEBUG ) std::cout << "Initializing Cell Fields\n";

            // Define device_type for Later Use
            using device_type = typename cell_array::device_type;

            // Get Local Grid and Local Mesh
            auto local_grid = *( _mesh->localGrid() );
            auto local_mesh = Cajita::createLocalMesh<device_type>( local_grid );

            // DEBUG: Print Low Corner and High Corner of Local Mesh
            if ( DEBUG ) std::cout << "Rank: " << _mesh->rank() << "\tLow Corner: " << \
            local_mesh.lowCorner( Cajita::Own(), 0 ) << local_mesh.lowCorner( Cajita::Own(), 1 ) << local_mesh.lowCorner( Cajita::Own(), 2 ) << "\n";
            if ( DEBUG ) std::cout << "Rank: " << _mesh->rank() << "\tHigh Corner: " << \
            local_mesh.highCorner( Cajita::Own(), 0 ) << local_mesh.highCorner( Cajita::Own(), 1 ) << local_mesh.highCorner( Cajita::Own(), 2 ) << "\n";

            // Get Ghost Cell and Owned Cells Index Spaces
            auto ghost_cells = local_grid.indexSpace( Cajita::Ghost(), Cajita::Cell(), Cajita::Local() );
            auto owned_cells = local_grid.indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );

            // Get State Arrays
            auto u_a = get(Location::Cell(), Field::Momentum(), 0 );
            auto h_a = get(Location::Cell(), Field::Height(), 0 );
            auto u_b = get(Location::Cell(), Field::Momentum(), 1 );
            auto h_b = get(Location::Cell(), Field::Height(), 1 );

            // Loop Over All Ghost Cells ( i, j, k )
            Kokkos::parallel_for( "Initializing", Cajita::createExecutionPolicy( ghost_cells, ExecutionSpace() ), KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                // DEBUG: Print Rank and Owned / Ghost Extents
                // if ( DEBUG ) std::cout << "Rank: " << _mesh->rank() << "\tOwned Extent: " << owned_cells.extent( 0 ) << owned_cells.extent( 1 ) << owned_cells.extent( 2 ) << "\n";
                // if ( DEBUG ) std::cout << "Rank: " << _mesh->rank() << "\tGhost Extent: " << ghost_cells.extent( 0 ) << ghost_cells.extent( 1 ) << ghost_cells.extent( 2 ) << "\n";
                
                // Initialize State Vectors
                state_t momentum[2];
                state_t height;

                // Get Coordinates Associated with Indices ( i, j, k )
                int coords[3] = { i, j, k };
                state_t x[3];

                local_mesh.coordinates( Cajita::Cell(), coords, x );

                // Initialization Function
                create_functor(coords, x, momentum, height);

                // DEBUG: Print Rank, Indices, Coordinates, and Initialized Momentums and Height
                // if ( DEBUG ) std::cout << "Rank: " << _mesh->rank() << "\ti: " << i << "\tj: " << j << "\tk: " << k << "\tx: " << x[0] << "\ty: " << x[1] << "\tz: " \
                << x[2] << "\tu: " << momentum[0] << "\tv: " << momentum[1] << "\th: " << height << "\n";

                // Assign Values to State Views
                u_a( i, j, k, 0 ) = momentum[0];
                u_a( i, j, k, 1 ) = momentum[1];
                h_a( i, j, k, 0 ) = height;

                u_b( i, j, k, 0 ) = momentum[0];
		        u_b( i, j, k, 1 ) = momentum[1];
                h_b( i, j, k, 0 ) = height;

            } );

        };

        /**
         * Return mesh
         * @return Returns Mesh object
         **/
        const std::shared_ptr<Mesh<MemorySpace, ExecutionSpace, state_t>>& mesh() const {
            return _mesh;
        };

        /**
         * Return Momentum Field
         * @param Location::Cell
         * @param Field::Momentum
         * @param t Toggle between momentum arrays
         * @return Returns Momentum state array at cell centers
         **/
        typename cell_array::view_type get( Location::Cell, Field::Momentum, int t ) const {
            if ( t == 0 ) return _momentum_a->view();
            else return _momentum_b->view();
        };

        /**
         * Return Height Field
         * @param Location::Cell
         * @param Field::Height
         * @param t Toggle between height arrays
         * @return Returns Height state array at cell centers
         **/
        typename cell_array::view_type get( Location::Cell, Field::Height, int t ) const {
            if (t == 0 ) return _height_a->view();
            else return _height_b->view();
        };

        /**
         * Return HxFluxPlus Field at Cell Centers
         * @param Location::Cell
         * @param Field::HxFluxPlus
         * @return Returns HxFluxPlus state array at cell centers
         **/
        typename cell_array::view_type get( Location::Cell, Field::HxFluxPlus ) const {
            return _hx_flux_plus->view();
        }

        /**
         * Return HxFluxMinus Field at Cell Centers
         * @param Location::Cell
         * @param Field::HxFluxMinus
         * @return Returns HxFluxMinus state array at cell centers
         **/
        typename cell_array::view_type get( Location::Cell, Field::HxFluxMinus ) const {
            return _hx_flux_minus->view();
        }

        /**
         * Return UxFluxPlus Field at Cell Centers
         * @param Location::Cell
         * @param Field::UxFluxPlus
         * @return Returns UxFluxPlus state array at cell centers
         **/
        typename cell_array::view_type get( Location::Cell, Field::UxFluxPlus ) const {
            return _ux_flux_plus->view();
        }

        /**
         * Return UxFluxMinus Field at Cell Centers
         * @param Location::Cell
         * @param Field::UxFluxMinus
         * @return Returns UxFluxMinus state array at cell centers
         **/
        typename cell_array::view_type get( Location::Cell, Field::UxFluxMinus ) const {
            return _ux_flux_minus->view();
        }

        /**
         * Return HyFluxPlus Field at Cell Centers
         * @param Location::Cell
         * @param Field::HyFluxPlus
         * @return Returns HyFluxPlus state array at cell centers
         **/
        typename cell_array::view_type get( Location::Cell, Field::HyFluxPlus ) const {
            return _hy_flux_plus->view();
        }

        /**
         * Return HyFluxMinus Field at Cell Centers
         * @param Location::Cell
         * @param Field::HyFluxMinus
         * @return Returns HyFluxMinus state array at cell centers
         **/
        typename cell_array::view_type get( Location::Cell, Field::HyFluxMinus ) const {
            return _hy_flux_minus->view();
        }

        /**
         * Return UyFluxPlus Field at Cell Centers
         * @param Location::Cell
         * @param Field::UyFluxPlus
         * @return Returns UyFluxPlus state array at cell centers
         **/
        typename cell_array::view_type get( Location::Cell, Field::UyFluxPlus ) const {
            return _uy_flux_plus->view();
        }

        /**
         * Return UyFluxMinus Field at Cell Centers
         * @param Location::Cell
         * @param Field::UyFluxMinus
         * @return Returns UyFluxMinus state array at cell centers
         **/
        typename cell_array::view_type get( Location::Cell, Field::UyFluxMinus ) const {
            return _uy_flux_minus->view();
        }

        /**
         * Return HxWPlus Field at Cell Centers
         * @param Location::Cell
         * @param Field::HxWPlus
         * @return Returns HxWPlus state array at cell centers
         **/
        typename cell_array::view_type get( Location::Cell, Field::HxWPlus ) const {
            return _hx_w_plus->view();
        }

        /**
         * Return HxWMinus Field at Cell Centers
         * @param Location::Cell
         * @param Field::HxWMinus
         * @return Returns HxWMinus state array at cell centers
         **/
        typename cell_array::view_type get( Location::Cell, Field::HxWMinus ) const {
            return _hx_w_minus->view();
        }

        /**
         * Return HyWPlus Field at Cell Centers
         * @param Location::Cell
         * @param Field::HyWPlus
         * @return Returns HyWPlus state array at cell centers
         **/
        typename cell_array::view_type get( Location::Cell, Field::HyWPlus ) const {
            return _hy_w_plus->view();
        }

        /**
         * Return HyWMinus Field at Cell Centers
         * @param Location::Cell
         * @param Field::HyWMinus
         * @return Returns HyWMinus state array at cell centers
         **/
        typename cell_array::view_type get( Location::Cell, Field::HyWMinus ) const {
            return _hy_w_minus->view();
        }

        /**
         * Return UWPlus Field at Cell Centers
         * @param Location::Cell
         * @param Field::UWPlus
         * @return Returns UWPlus state array at cell centers
         **/
        typename cell_array::view_type get( Location::Cell, Field::UWPlus ) const {
            return _u_w_plus->view();
        }

        /**
         * Return UWMinus Field at Cell Centers
         * @param Location::Cell
         * @param Field::UWMinus
         * @return Returns UWMinus state array at cell centers
         **/
        typename cell_array::view_type get( Location::Cell, Field::UWMinus ) const {
            return _u_w_minus->view();
        }

        /**
         * Scatter Momentum to Neighbors
         * @param Location::Cell
         * @param Field::Momentum
         * @param t Toggle between momentum arrays
         **/
        void scatter( Location::Cell, Field::Momentum, int t ) const {
            if ( t == 0 ) _cell_vector_halo->scatter( *_momentum_a );
            else _cell_vector_halo->scatter( *_momentum_b );
        };

        /**
         * Scatter Height to Neighbors
         * @param Location::Cell
         * @param Field::Height
         * @param t Toggle between height arrays
         **/
        void scatter( Location::Cell, Field::Height, int t ) const {
            if ( t == 0 ) _cell_scalar_halo->scatter( *_height_a );
            else _cell_scalar_halo->scatter( *_height_b );
        };

        /**
         * Gather Momentum from Neighbors
         * @param Location::Cell
         * @param Field::Momentum
         * @param t Toggle between momentum arrays
         **/
        void gather( Location::Cell, Field::Momentum, int t ) const {
            if ( t == 0 ) _cell_vector_halo->gather( *_momentum_a );
            else _cell_vector_halo->gather( *_momentum_b );
        };

        /**
         * Gather Height from Neighbors
         * @param Location::Cell
         * @param Field::Height
         * @param t Toggle between height arrays
         **/
        void gather( Location::Cell, Field::Height, int t ) const {
            if ( t == 0 ) _cell_scalar_halo->gather( *_height_a );
            else _cell_scalar_halo->gather( *_height_b );
        };

        /**
         * Gather Height and Momentum from Neighbors if using Cuda UVM
         * @param mindt Time step (dt)
         * @param time_Step Current time step
         **/
        void gatherCuda( state_t mindt, int time_step ) const {
            auto local_grid = mesh()->localGrid();
            auto owned_cells = local_grid->indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );

            auto uCurrent = get( Location::Cell(), Field::Momentum(), CURRENTFIELD( time_step ) );
            auto hCurrent = get( Location::Cell(), Field::Height(), CURRENTFIELD( time_step ) );

            auto uNew = get( Location::Cell(), Field::Momentum(), NEWFIELD( time_step ) );
            auto hNew = get( Location::Cell(), Field::Height(), NEWFIELD( time_step ) );

            for ( int i = -1; i < 2; i++ ) {
                for (int j = -1; j < 2; j++) {
                    if ( ( i == 0 || j == 0 ) && !( i == 0 && j == 0 ) ){
                        int neighbor = local_grid->neighborRank( i, j, 0 );
                        if ( neighbor != -1 ) {
                            auto shared_recv_cells = local_grid->sharedIndexSpace( Cajita::Ghost(), Cajita::Cell(), i, j, 0 );
                            auto shared_send_cells = local_grid->sharedIndexSpace( Cajita::Own(), Cajita::Cell(), i, j, 0 );

                            // DEBUG: Print neighbor information
                            /*
                            if ( DEBUG ) std::cout << "Rank: " << pm.mesh()->rank() << "\t i: " << i << "\tj: " << j << "\tk: " << 0 << "\tNeighbor: " << neighbor << "\n";
                            if ( DEBUG ) std::cout << "Rank (Recv): " << pm.mesh()->rank() << "\txmin: " << shared_recv_cells.min( 0 ) << "\txmax: " << shared_recv_cells.max( 0 ) \
                            << "\tymin: " << shared_recv_cells.min( 1 ) << "\tymax: " << shared_recv_cells.max( 1 ) << "\tzmin: " << shared_recv_cells.min( 2 ) << "\tzmax: " << shared_recv_cells.max( 2 ) << "\n";
                            if ( DEBUG ) std::cout << "Rank (Send): " << pm.mesh()->rank() << "\txmin: " << shared_send_cells.min( 0 ) << "\txmax: " << shared_send_cells.max( 0 ) \
                            << "\tymin: " << shared_send_cells.min( 1 ) << "\tymax: " << shared_send_cells.max( 1 ) << "\tzmin: " << shared_send_cells.min( 2 ) << "\tzmax: " << shared_send_cells.max( 2 ) << "\n";
                            */

                            state_t sendH[shared_send_cells.size()];
                            state_t sendU[shared_send_cells.size()];
                            state_t sendV[shared_send_cells.size()];
                            state_t recvH[shared_recv_cells.size()];
                            state_t recvU[shared_recv_cells.size()];
                            state_t recvV[shared_recv_cells.size()];

                            for ( int ii = shared_send_cells.min( 0 ); ii < shared_send_cells.max( 0 ); ii++ ) {
                                for ( int jj = shared_send_cells.min( 1 ); jj < shared_send_cells.max( 1 ); jj++ ) {
                                    for ( int kk = shared_send_cells.min( 2 ); kk < shared_send_cells.max( 2 ); kk++ ) {
                                        int ii_own = ii - shared_send_cells.min( 0 );
                                        int jj_own = jj - shared_send_cells.min( 1 );
                                        int kk_own = kk - shared_send_cells.min( 2 );

                                        int inx = ii_own + shared_send_cells.extent( 0 ) * ( jj_own + shared_send_cells.extent( 1 ) * kk_own );

                                        // DEBUG: Print Index Conversion
                                        // if ( DEBUG ) std::cout << "Rank: " << pm.mesh()->rank() << "ii: " << ii << "\tjj: " << jj << "\tkk: " << kk \
                                        << "\t ii_own: " << ii_own << "\tjj_own: " << jj_own << "\tkk_own: " << kk_own << "\tinx: " << inx << "\n";

                                        sendH[inx] = hNew( ii, jj, kk, 0 );
                                        sendU[inx] = uNew( ii, jj, kk, 0 );
                                        sendV[inx] = uNew( ii, jj, kk, 1 );
                                    }
                                }
                            }

                            MPI_Request request[6];
                            MPI_Status statuses[6];

                            MPI_Isend( sendH, shared_send_cells.size(), Cajita::MpiTraits<state_t>::type(), neighbor, 0, MPI_COMM_WORLD, &request[0] );
                            MPI_Isend( sendU, shared_send_cells.size(), Cajita::MpiTraits<state_t>::type(), neighbor, 0, MPI_COMM_WORLD, &request[1] );
                            MPI_Isend( sendV, shared_send_cells.size(), Cajita::MpiTraits<state_t>::type(), neighbor, 0, MPI_COMM_WORLD, &request[2] );

                            MPI_Irecv( recvH, shared_recv_cells.size(), Cajita::MpiTraits<state_t>::type(), neighbor, 0, MPI_COMM_WORLD, &request[3] );
                            MPI_Irecv( recvU, shared_recv_cells.size(), Cajita::MpiTraits<state_t>::type(), neighbor, 0, MPI_COMM_WORLD, &request[4] );
                            MPI_Irecv( recvV, shared_recv_cells.size(), Cajita::MpiTraits<state_t>::type(), neighbor, 0, MPI_COMM_WORLD, &request[5] );

                            MPI_Waitall( 6, request, statuses );
                            
                            for ( int ii = shared_recv_cells.min( 0 ); ii < shared_recv_cells.max( 0 ); ii++ ) {
                                for ( int jj = shared_recv_cells.min( 1 ); jj < shared_recv_cells.max( 1 ); jj++ ) {
                                    for ( int kk = shared_recv_cells.min( 2 ); kk < shared_recv_cells.max( 2 ); kk++ ) {
                                        int ii_own = ii - shared_recv_cells.min( 0 );
                                        int jj_own = jj - shared_recv_cells.min( 1 );
                                        int kk_own = kk - shared_recv_cells.min( 2 );

                                        int inx = ii_own + shared_recv_cells.extent( 0 ) * ( jj_own + shared_recv_cells.extent( 1 ) * kk_own );

                                        // DEBUG: Print index conversion and received data
                                        // if ( DEBUG ) std::cout << "Rank: " << pm.mesh()->rank() << "\tii: " << ii << "\tjj: " << jj << "\tkk: " << kk \
                                        << "\t ii_own: " << ii_own << "\tjj_own: " << jj_own << "\tkk_own: " << kk_own << "\tinx: " << inx << "\trecvH: " << recvH[inx] << "\n";

                                        hNew( ii, jj, kk, 0 ) = recvH[inx];
                                        uNew( ii, jj, kk, 0 ) = recvU[inx];
                                        uNew( ii, jj, kk, 1 ) = recvV[inx];
                                    }
                                }
                            }
                        }
                    }
                }   
            }
        };

    private:
#if 0
        Cabana::AoSoA<cell_members, MemorySpace> _cells;
#endif
        std::shared_ptr<Mesh<MemorySpace, ExecutionSpace, state_t>> _mesh;      /**< Mesh object */
        std::shared_ptr<cell_array> _momentum_a;                                /**< Momentum state array 1 */
        std::shared_ptr<cell_array> _height_a;                                  /**< Height state array 1 */
        std::shared_ptr<cell_array> _momentum_b;                                /**< Momentum state array 2 */
        std::shared_ptr<cell_array> _height_b;                                  /**< Height state array 2 */

        std::shared_ptr<cell_array> _hx_flux_plus;                              /**< Height x-direction positive flux array */
        std::shared_ptr<cell_array> _hx_flux_minus;                             /**< Height x-direction negative flux array */
        std::shared_ptr<cell_array> _ux_flux_plus;                              /**< X-Momentum positive flux array */
        std::shared_ptr<cell_array> _ux_flux_minus;                             /**< X-Momentum negative flux array */

        std::shared_ptr<cell_array> _hy_flux_plus;                              /**< Height y-direction positive flux array */
        std::shared_ptr<cell_array> _hy_flux_minus;                             /**< Height y-direction negative flux array */
        std::shared_ptr<cell_array> _uy_flux_plus;                              /**< Y-Momentum positive flux array */
        std::shared_ptr<cell_array> _uy_flux_minus;                             /**< Y-Momentum negative flux array */

        std::shared_ptr<cell_array> _hx_w_plus;                                 /**< Height x-direction positive flux corrector array */
        std::shared_ptr<cell_array> _hx_w_minus;                                /**< Height x-direction negative flux corrector array */
        std::shared_ptr<cell_array> _hy_w_plus;                                 /**< Height y-direction positive flux corrector array */
        std::shared_ptr<cell_array> _hy_w_minus;                                /**< Height y-direction negative flux corrector array */

        std::shared_ptr<cell_array> _u_w_plus;                                  /**< Momentum positive flux corrector array */
        std::shared_ptr<cell_array> _u_w_minus;                                 /**< Momentum negative flux corrector array */

        std::shared_ptr<halo> _cell_vector_halo;                                /**< Halo for vector arrays */
        std::shared_ptr<halo> _cell_scalar_halo;                                /**< Halo for scalar arrays */
};

}

#endif

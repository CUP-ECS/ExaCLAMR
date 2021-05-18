/**
 * @file
 * @author Patrick Bridges <patrickb@unm.edu>
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
#include <ExaClamrTypes.hpp>
#include <Mesh.hpp>

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <memory>

namespace ExaCLAMR {

    /**
 * @namespace Location
 * @brief Location namespace to track mesh entities
 **/
    namespace Location {
        /**
 * @struct Cell
 * @brief Cell Location Type 
 **/
        struct Cell {};
    } // namespace Location

    /**
 * @namespace Field
 * @brief Field namespace to track state array entities
 **/
    namespace Field {
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
    } // namespace Field

    /**
 * The ProblemManager Class
 * @class ProblemManager
 * @brief ProblemManager class to store the mesh and state values, and to perform gathers and scatters.
 **/
    template <class MeshType, class MemorySpace, class ExecutionSpace, class OrderingView>
    class ProblemManager;

    template <class state_t, class MemorySpace, class ExecutionSpace, class OrderingView>
    class ProblemManager<ExaCLAMR::AMRMesh<state_t>, MemorySpace, ExecutionSpace, OrderingView> {
        using cell_members = Cabana::MemberTypes<state_t[3], state_t[3], state_t[3][3]>;

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
        ProblemManager( const ExaCLAMR::ClArgs<state_t> &cl, const Cajita::ManualPartitioner &partitioner, MPI_Comm comm, const InitFunc &create_functor ) {
            // Create Mesh
            _mesh = std::make_shared<Mesh<ExaCLAMR::AMRMesh<state_t>, MemorySpace>>( cl, partitioner, comm );

            // trace Create Problem Manager
            if ( DEBUG && _mesh->rank() == 0 ) std::cout << "Created AMR ProblemManager\n";

            initialize( create_functor );
        }

        /**
         * Initializes state values in the cells
         * @param create_functor Initialization function
         **/
        template <class InitFunctor>
        void initialize( const InitFunctor &create_functor ){

        };

        /**
         * Return mesh
         * @return Returns Mesh object
         **/
        const std::shared_ptr<Mesh<ExaCLAMR::AMRMesh<state_t>, MemorySpace>> &mesh() const {
            return _mesh;
        };

      private:
        Cabana::AoSoA<cell_members, MemorySpace> _cells;

        std::shared_ptr<Mesh<ExaCLAMR::AMRMesh<state_t>, MemorySpace>> _mesh; /**< Mesh object */
    };

    template <class state_t, class MemorySpace, class ExecutionSpace, class OrderingView>
    class ProblemManager<ExaCLAMR::RegularMesh<state_t>, MemorySpace, ExecutionSpace, OrderingView> {
        using cell_array  = Cajita::Array<state_t, Cajita::Cell, Cajita::UniformMesh<state_t>, OrderingView, MemorySpace>;
        using halo        = Cajita::Halo<MemorySpace>;
        using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

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
        ProblemManager( const ExaCLAMR::ClArgs<state_t> &cl, const Cajita::ManualPartitioner &partitioner, MPI_Comm comm, const InitFunc &create_functor ) {
            // Create Mesh
            _mesh = std::make_shared<Mesh<ExaCLAMR::RegularMesh<state_t>, MemorySpace>>( cl, partitioner, comm );

            // Trace Create Problem Manager
            if ( DEBUG && _mesh->rank() == 0 ) std::cout << "Created Regular ProblemManager\n";

            // Create Vector and Scalar Layouts
            auto cell_vector_layout = Cajita::createArrayLayout( _mesh->localGrid(), 2, Cajita::Cell() ); // 2-Dimensional ( Momentum )
            auto cell_scalar_layout = Cajita::createArrayLayout( _mesh->localGrid(), 1, Cajita::Cell() );

            // Initialize State Arrays
            // A and B Arrays used to Update State Data without Overwriting
            _momentum_a = Cajita::createArray<state_t, OrderingView, MemorySpace>( "momentum", cell_vector_layout );
            _height_a   = Cajita::createArray<state_t, OrderingView, MemorySpace>( "height", cell_scalar_layout );

            _momentum_b = Cajita::createArray<state_t, OrderingView, MemorySpace>( "momentum", cell_vector_layout );
            _height_b   = Cajita::createArray<state_t, OrderingView, MemorySpace>( "height", cell_scalar_layout );

            // Create Halo Pattern
            auto                            halo_pattern = Cajita::HaloPattern<3>();
            std::vector<std::array<int, 3>> neighbors;

            // Setting up Stencil ( Left, Right, Top, Bottom )
            for ( int i = -1; i < 2; i++ ) {
                for ( int j = -1; j < 2; j++ ) {
                    if ( ( i == 0 || j == 0 ) && !( i == 0 && j == 0 ) ) {
                        neighbors.push_back( { i, j, 0 } );
                    }
                }
            }

            halo_pattern.setNeighbors( neighbors );

            // Initialize Halo Array Layours
            _cell_state_halo = Cajita::createHalo( halo_pattern, cl.halo_size, *_momentum_a, *_height_a, *_momentum_b, *_height_b );

            // Initialize State Values ( Height, Momentum )
            initialize( create_functor );
        };

        /**
         * Initializes state values in the cells
         * @param create_functor Initialization function
         **/
        template <class InitFunctor>
        void initialize( const InitFunctor &create_functor ) {
            // DEBUG: Trace State Initialization
            if ( _mesh->rank() == 0 && DEBUG ) std::cout << "Initializing Cell Fields\n";

            // Get Local Grid and Local Mesh
            auto local_grid = *( _mesh->localGrid() );
            auto local_mesh = Cajita::createLocalMesh<device_type>( local_grid );

            // DEBUG: Print Low Corner and High Corner of Local Mesh
            if ( DEBUG ) std::cout << "Rank: " << _mesh->rank() << "\tLow Corner: " << local_mesh.lowCorner( Cajita::Own(), 0 ) << local_mesh.lowCorner( Cajita::Own(), 1 ) << local_mesh.lowCorner( Cajita::Own(), 2 ) << "\n";
            if ( DEBUG ) std::cout << "Rank: " << _mesh->rank() << "\tHigh Corner: " << local_mesh.highCorner( Cajita::Own(), 0 ) << local_mesh.highCorner( Cajita::Own(), 1 ) << local_mesh.highCorner( Cajita::Own(), 2 ) << "\n";

            // Get Ghost Cell Index Space
            auto ghost_cells = local_grid.indexSpace( Cajita::Ghost(), Cajita::Cell(), Cajita::Local() );

            // Get State Arrays
            auto u_a = get( Location::Cell(), Field::Momentum(), 0 );
            auto h_a = get( Location::Cell(), Field::Height(), 0 );
            auto u_b = get( Location::Cell(), Field::Momentum(), 1 );
            auto h_b = get( Location::Cell(), Field::Height(), 1 );

            // Loop Over All Ghost Cells ( i, j, k )
            Kokkos::parallel_for(
                "Initializing", Cajita::createExecutionPolicy( ghost_cells, ExecutionSpace() ), KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                    // DEBUG: Print Rank and Owned / Ghost Extents
                    // if ( DEBUG ) std::cout << "Rank: " << _mesh->rank() << "\tOwned Extent: " << owned_cells.extent( 0 ) << owned_cells.extent( 1 ) << owned_cells.extent( 2 ) << "\n";
                    // if ( DEBUG ) std::cout << "Rank: " << _mesh->rank() << "\tGhost Extent: " << ghost_cells.extent( 0 ) << ghost_cells.extent( 1 ) << ghost_cells.extent( 2 ) << "\n";

                    // Initialize State Vectors
                    state_t momentum[2];
                    state_t height;

                    // Get Coordinates Associated with Indices ( i, j, k )
                    int     coords[3] = { i, j, k };
                    state_t x[3];

                    local_mesh.coordinates( Cajita::Cell(), coords, x );

                    // Initialization Function
                    create_functor( coords, x, momentum, height );

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
        const std::shared_ptr<Mesh<ExaCLAMR::RegularMesh<state_t>, MemorySpace>> &mesh() const {
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
            if ( t == 0 )
                return _momentum_a->view();
            else
                return _momentum_b->view();
        };

        /**
         * Return Height Field
         * @param Location::Cell
         * @param Field::Height
         * @param t Toggle between height arrays
         * @return Returns Height state array at cell centers
         **/
        typename cell_array::view_type get( Location::Cell, Field::Height, int t ) const {
            if ( t == 0 )
                return _height_a->view();
            else
                return _height_b->view();
        };

        /**
         * Scatter State Data to Neighbors
         * @param Location::Cell
         * @param t Toggle between state arrays
         **/
        void scatter( Location::Cell, int t ) const {
            if ( t == 0 )
                _cell_state_halo->scatter( ExecutionSpace(), *_momentum_a, *_height_a );
            else
                _cell_state_halo->scatter( ExecutionSpace(), *_momentum_b, *_height_b );
        };

        /**
         * Gather State Data from Neighbors
         * @param Location::Cell
         * @param t Toggle between state arrays
         **/
        void gather( Location::Cell, int t ) const {
            if ( t == 0 )
                _cell_state_halo->gather( ExecutionSpace(), *_momentum_a, *_height_a );
            else
                _cell_state_halo->gather( ExecutionSpace(), *_momentum_b, *_height_b );
        }

      private:
        std::shared_ptr<Mesh<ExaCLAMR::RegularMesh<state_t>, MemorySpace>> _mesh; /**< Mesh object */

        std::shared_ptr<cell_array> _momentum_a; /**< Momentum state array 1 */
        std::shared_ptr<cell_array> _height_a;   /**< Height state array 1 */
        std::shared_ptr<cell_array> _momentum_b; /**< Momentum state array 2 */
        std::shared_ptr<cell_array> _height_b;   /**< Height state array 2 */

        std::shared_ptr<halo> _cell_state_halo; /**< Halo for A state arrays */
    };

} // namespace ExaCLAMR

#endif

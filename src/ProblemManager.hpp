/**
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * 
 */

#ifndef EXACLAMR_PROBLEMMANAGER_HPP
#define EXACLAMR_PROBLEMMANAGER_HPP

#ifndef DEBUG
    #define DEBUG 0 
#endif 

#include <Mesh.hpp>

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <memory>

namespace ExaCLAMR
{

// Location Struct
namespace Location
{
struct Cell {};
struct Face {};
struct Node {};
}

// Fields Struct
namespace Field
{
// TODO: Change from Velocity to Momentum
struct Velocity {};
struct Height {};

struct HxFluxPlus {};
struct HxFluxMinus {};
struct UxFluxPlus {};
struct UxFluxMinus {};

struct HyFluxPlus {};
struct HyFluxMinus {};
struct UyFluxPlus {};
struct UyFluxMinus {};

struct HxWPlus {};
struct HxWMinus {};
struct HyWPlus {};
struct HyWMinus {};

struct UWPlus {};
struct UWMinus {};
}

template<class MemorySpace, class ExecutionSpace, typename state_t>
class ProblemManager
{
    using cell_array = Cajita::Array<state_t, Cajita::Cell, Cajita::UniformMesh<state_t>, MemorySpace>;
    using halo = Cajita::Halo<state_t, MemorySpace>;

    public:
        /**
         * Constructor
         * 
         * @param
         */
        template <class InitFunc>
        ProblemManager( const std::array<state_t, 6>& global_bounding_box, 
                const std::array<int, 3>& global_num_cell,
                const std::array<bool, 3>& periodic,
                const Cajita::Partitioner& partitioner, 
                const int halo_size, 
                MPI_Comm comm,
                const InitFunc& create_functor ) {
            // Create Mesh
            _mesh = std::make_shared<Mesh<MemorySpace, ExecutionSpace, state_t>> ( 
                                            global_bounding_box, 
                                            global_num_cell, 
                                            periodic, 
                                            partitioner, 
                                            halo_size, 
                                            comm );

            // Create Vector and Scalar Layouts
            auto cell_vector_layout = Cajita::createArrayLayout( _mesh->localGrid(), 2, Cajita::Cell() );   // 2-Dimensional ( Velocity / Momentum )
            auto cell_scalar_layout = Cajita::createArrayLayout( _mesh->localGrid(), 1, Cajita::Cell() );

            // Initialize State Arrays
            // A and B Arrays used to Update State Data without Overwriting
            _velocity_a = Cajita::createArray<state_t, MemorySpace>( "velocity", cell_vector_layout );
            _height_a = Cajita::createArray<state_t, MemorySpace>( "height", cell_scalar_layout );

            _velocity_b = Cajita::createArray<state_t, MemorySpace>( "velocity", cell_vector_layout );
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

            // Initialize State Values ( Height, Velocity/Momentum )
            initialize( create_functor );

        };

        template<class InitFunctor>
        void initialize( const InitFunctor& create_functor ) {
            if ( _mesh->rank() == 0 && DEBUG ) std::cout << "Initializing Cell Fields\n";       // DEBUG: Trace State Initialization

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
	        auto u_a = get(Location::Cell(), Field::Velocity(), 0 );
	        auto h_a = get(Location::Cell(), Field::Height(), 0 );
            auto u_b = get(Location::Cell(), Field::Velocity(), 1 );
	        auto h_b = get(Location::Cell(), Field::Height(), 1 );

            // Loop Over All Ghost Cells ( i, j, k )
            Kokkos::parallel_for( "Initializing", Cajita::createExecutionPolicy( ghost_cells, ExecutionSpace() ), KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                // DEBUG: Print Rank and Owned / Ghost Extents
                if ( DEBUG ) std::cout << "Rank: " << _mesh->rank() << "\tOwned Extent: " << owned_cells.extent( 0 ) << owned_cells.extent( 1 ) << owned_cells.extent( 2 ) << "\n";
                if ( DEBUG ) std::cout << "Rank: " << _mesh->rank() << "\tGhost Extent: " << ghost_cells.extent( 0 ) << ghost_cells.extent( 1 ) << ghost_cells.extent( 2 ) << "\n";
                
                // Initialize State Vectors
                state_t velocity[2];
                state_t height;

                // Get Coordinates Associated with Indices ( i, j, k )
                int coords[3] = { i, j, k };
                state_t x[3];

                local_mesh.coordinates( Cajita::Cell(), coords, x );

                // Initialization Function
                create_functor(coords, x, velocity, height);

                // DEBUG: Print Rank, Indices, Coordinates, and Initialized Velocities and Height
                if ( DEBUG ) std::cout << "Rank: " << _mesh->rank() << "\ti: " << i << "\tj: " << j << "\tk: " << k << "\tx: " << x[0] << "\ty: " << x[1] << "\tz: " \
                << x[2] << "\tvx: " << velocity[0] << "\tvy: " << velocity[1] << "\th: " << height << "\n";

                // Assign Values to State Views
		        u_a( i, j, k, 0 ) = velocity[0];
		        u_a( i, j, k, 1 ) = velocity[1];
                h_a( i, j, k, 0 ) = height;

                u_b( i, j, k, 0 ) = velocity[0];
		        u_b( i, j, k, 1 ) = velocity[1];
                h_b( i, j, k, 0 ) = height;

            } );

        };

        const std::shared_ptr<Mesh<MemorySpace, ExecutionSpace, state_t>>& mesh() const {
            return _mesh;
        };

        typename cell_array::view_type get( Location::Cell, Field::Velocity, int t ) const {
            if ( t == 0 ) return _velocity_a->view();
            else return _velocity_b->view();
        };

        typename cell_array::view_type get( Location::Cell, Field::Height, int t ) const {
            if (t == 0 ) return _height_a->view();
            else return _height_b->view();
        };

        typename cell_array::view_type get( Location::Cell, Field::HxFluxPlus ) const {
            return _hx_flux_plus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::HxFluxMinus ) const {
            return _hx_flux_minus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::UxFluxPlus ) const {
            return _ux_flux_plus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::UxFluxMinus ) const {
            return _ux_flux_minus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::HyFluxPlus ) const {
            return _hy_flux_plus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::HyFluxMinus ) const {
            return _hy_flux_minus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::UyFluxPlus ) const {
            return _uy_flux_plus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::UyFluxMinus ) const {
            return _uy_flux_minus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::HxWPlus ) const {
            return _hx_w_plus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::HxWMinus ) const {
            return _hx_w_minus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::HyWPlus ) const {
            return _hy_w_plus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::HyWMinus ) const {
            return _hy_w_minus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::UWPlus ) const {
            return _u_w_plus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::UWMinus ) const {
            return _u_w_minus->view();
        }

        void scatter( Location::Cell, Field::Velocity, int t ) const {
            if ( t == 0 ) _cell_vector_halo->scatter( *_velocity_a );
            else _cell_vector_halo->scatter( *_velocity_b );
        };

        void scatter( Location::Cell, Field::Height, int t ) const {
            if ( t == 0 ) _cell_scalar_halo->scatter( *_height_a );
            else _cell_scalar_halo->scatter( *_height_b );
        };

        void gather( Location::Cell, Field::Velocity, int t ) const {
            if ( t == 0 ) _cell_vector_halo->gather( *_velocity_a );
            else _cell_vector_halo->gather( *_velocity_b );
        }

        void gather( Location::Cell, Field::Height, int t ) const {
            if ( t == 0 ) _cell_scalar_halo->gather( *_height_a );
            else _cell_scalar_halo->gather( *_height_b );
        }

    private:
#if 0
        Cabana::AoSoA<cell_members, MemorySpace> _cells;
#endif
        std::shared_ptr<Mesh<MemorySpace, ExecutionSpace, state_t>> _mesh;
        std::shared_ptr<cell_array> _velocity_a;
        std::shared_ptr<cell_array> _height_a;
        std::shared_ptr<cell_array> _velocity_b;
        std::shared_ptr<cell_array> _height_b;

        std::shared_ptr<cell_array> _hx_flux_plus;
        std::shared_ptr<cell_array> _hx_flux_minus;
        std::shared_ptr<cell_array> _ux_flux_plus;
        std::shared_ptr<cell_array> _ux_flux_minus;

        std::shared_ptr<cell_array> _hy_flux_plus;
        std::shared_ptr<cell_array> _hy_flux_minus;
        std::shared_ptr<cell_array> _uy_flux_plus;
        std::shared_ptr<cell_array> _uy_flux_minus;

        std::shared_ptr<cell_array> _hx_w_plus;
        std::shared_ptr<cell_array> _hx_w_minus;
        std::shared_ptr<cell_array> _hy_w_plus;
        std::shared_ptr<cell_array> _hy_w_minus;

        std::shared_ptr<cell_array> _u_w_plus;
        std::shared_ptr<cell_array> _u_w_minus;

        std::shared_ptr<halo> _cell_vector_halo;
        std::shared_ptr<halo> _cell_scalar_halo;
};

}

#endif


/**
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * 
 */


namespace ExaCLAMR
{
// Toggle Between Current and New State Vectors
#define NEWFIELD( time_step ) ( ( time_step + 1 ) % 2 )
#define CURRENTFIELD( time_step ) ( ( time_step ) % 2 )

} 
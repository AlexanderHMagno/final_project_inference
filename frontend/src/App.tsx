import { ChakraProvider, Container, Heading, Box, Text, useColorMode, VStack } from '@chakra-ui/react'
// import { SunIcon, MoonIcon } from '@chakra-ui/icons'
import ImageUploader from './components/ImageUploader'

function App() {
  const { colorMode } = useColorMode()

  return (
    <ChakraProvider>
      <Box 
        minH="100vh"
        minW="100vw"
        bg={colorMode === 'light' ? 'gray.50' : 'gray.900'}
        transition="background 0.2s"
      >
        
        <Container maxW="container.sm" py={12}>
          <VStack spacing={8}>
            <Box textAlign="center">
              <Heading 
                as="h1" 
                size="2xl" 
                bgGradient="linear(to-r, blue.400, purple.500)" 
                bgClip="text"
                fontWeight="extrabold"
              >
                Person Detection
              </Heading>
              <Text 
                mt={4} 
                fontSize="lg" 
                color={colorMode === 'light' ? 'gray.600' : 'gray.400'}
              >
                Upload an image to detect people using YOLOv11
              </Text>
            </Box>


          </VStack>
        </Container>
        <Container maxW="container.xl" py={12}>
          <Box 
            w="full"
            bg={colorMode === 'light' ? 'white' : 'gray.800'}
            p={8}
            borderRadius="xl"
            boxShadow="xl"
            >
              <ImageUploader />
            </Box>
          </Container>
      </Box>
    </ChakraProvider>
  )
}

export default App

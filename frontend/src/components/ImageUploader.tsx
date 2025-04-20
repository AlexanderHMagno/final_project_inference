import { useState } from 'react'
import {
  Box,
  Button,
  Image,
  Text,
  Container,
  Flex,
  Spinner,
  useToast
} from '@chakra-ui/react'
import axios from 'axios'
import tiff from 'tiff.js'
import Swal from 'sweetalert2'

const API_URL = 'http://localhost:8000'

const ImageUploader = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [results, setResults] = useState<{
    patches: string | null;
    result: string | null;
  }>({ patches: null, result: null })
  
  const toast = useToast()

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      const reader = new FileReader()
      reader.onloadend = () => {

        // Check if file is TIFF
        if (file.type === 'image/tiff') {
          // Create new Tiff instance directly
          const tiffInstance = new tiff({buffer: reader.result as ArrayBuffer})
          const canvas = tiffInstance.toCanvas()
          const pngUrl = canvas.toDataURL('image/png')
          setPreviewUrl(pngUrl)
          
          // Convert base64 PNG to File object for upload
          const base64Data = pngUrl.split(',')[1]
          const bytes = atob(base64Data)
          const mime = 'image/png'
          const buffer = new ArrayBuffer(bytes.length)
          const arr = new Uint8Array(buffer)
          for (let i = 0; i < bytes.length; i++) {
            arr[i] = bytes.charCodeAt(i)
          }
          const blob = new Blob([arr], {type: mime})
          const convertedFile = new File([blob], file.name.replace('.tiff','.png'), {type: mime})
          setSelectedFile(convertedFile)
          return
        }
        setPreviewUrl(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const resetPatchesAndResult = () => {
    setResults({
      patches: null,
      result: null,
    })
  }

  
  const handleUpload = async () => {
    resetPatchesAndResult()

    if (!selectedFile) {
      toast({
        title: 'No file selected',
        status: 'error',
        duration: 3000,
        isClosable: true,
      })
      return
    }

    setIsLoading(true)

    // Show processing alert
    Swal.fire({
      title: 'Processing Your Image!',
      html: 'Our awesome AI model is analyzing your image using state-of-the-art YOLOv11 technology. This may take a few moments...',
      icon: 'info',
      allowOutsideClick: false,
      showConfirmButton: false,
      didOpen: () => {
        Swal.showLoading()
      }
    })
    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await axios.post(`${API_URL}/detect`, formData, {
        responseType: 'json',
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      setResults({
        patches: response.data.patches,
        result: response.data.result,
      })

      
    } catch (error) {
      console.error('Error uploading image:', error)
      toast({
        title: 'Error processing image',
        description: 'Please try again',
        status: 'error',
        duration: 3000,
        isClosable: true,
      })
    } finally {
      setIsLoading(false)
      Swal.close()
    }
  }

  return (
    <Container maxW="100%" p={0}>
      <Flex direction="column" justifyContent="space-between" gap={6}>
        <Box w="100%">
          <Flex direction="row" justifyContent="center" gap={4}>
            <input
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
                id="file-upload"
            />
            <label htmlFor="file-upload">
              <Button as="span" colorScheme="blue" size={"lg"}>
                Select Image
              </Button>
            </label>
          </Flex>
          {previewUrl && (
            <Box>
              <Text mb={2}>Selected Image:</Text>
              <Image src={previewUrl} maxH="1000px" objectFit="contain" />
            </Box>
          )}
        </Box>


        <Box my={10}>
        <Flex direction="row" justifyContent="center" gap={4}>
          <Button
              colorScheme="green"
              visibility={!selectedFile ? 'hidden' : 'visible'}
              onClick={handleUpload}
              size={"lg"}
          >
            {isLoading ? <Spinner size="sm" color="white" mr={2} /> : null}
            {isLoading ? 'Processing...' : 'Process Image'}
          </Button>
        </Flex>

        {(results.patches || results.result) && (
          <Flex direction="column" gap={4} py={10}>
            {results.patches && (
              <Box>
                <Text color="gray" fontSize="2xl" mb={5}>Patch Analysis:</Text>
                <Image src={results.patches} maxH="1000px" m="auto" objectFit="contain" />
              </Box>
            )}
            {results.result && (
              <Box>
                <Text color="gray.500" fontSize="2xl" mb={5}>Final Detection:</Text>
                <Image src={results.result} maxH="1000px" objectFit="contain" />
              </Box>
            )}
          </Flex>
        )}
        </Box>

      </Flex>
    </Container>
  )
}

export default ImageUploader 
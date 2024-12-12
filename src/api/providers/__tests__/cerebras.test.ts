import { CerebrasHandler } from '../cerebras'
import { ApiHandlerOptions, ModelInfo } from '../../../shared/api'
import { Anthropic } from '@anthropic-ai/sdk'
import axios from 'axios'

// Mock dependencies
jest.mock('axios')
jest.mock('delay', () => jest.fn(() => Promise.resolve()))

describe('CerebrasHandler', () => {
    const mockOptions: ApiHandlerOptions = {
        cerebrasApiKey: 'test-key',
        cerebrasModelId: 'llama-3.3-70b',
        cerebrasModelInfo: {
            name: 'Llama 3.3 70B',
            description: 'Llama 3.3 70B model from Cerebras - Latest version with improved performance',
            maxTokens: 4096,
            contextWindow: 8192,
            supportsPromptCache: true,
            inputPrice: 0.0001,
            outputPrice: 0.0002
        } as ModelInfo
    }

    beforeEach(() => {
        jest.clearAllMocks()
    })

    test('constructor initializes with correct options', () => {
        const handler = new CerebrasHandler(mockOptions)
        expect(handler).toBeInstanceOf(CerebrasHandler)
    })

    test('getModel returns correct model info when options are provided', () => {
        const handler = new CerebrasHandler(mockOptions)
        const result = handler.getModel()
        
        expect(result).toEqual({
            id: mockOptions.cerebrasModelId,
            info: mockOptions.cerebrasModelInfo
        })
    })

    test('createMessage generates correct stream chunks', async () => {
        const handler = new CerebrasHandler(mockOptions)
        const mockResponse = {
            choices: [{
                message: {
                    content: 'test response'
                }
            }]
        }

        // Mock axios.post for chat completion
        ;(axios.post as jest.Mock).mockResolvedValue({
            data: mockResponse
        })

        const systemPrompt = 'test system prompt'
        const messages: Anthropic.Messages.MessageParam[] = [
            { role: 'user', content: 'test message' }
        ]

        const generator = handler.createMessage(systemPrompt, messages)
        const chunks = []
        
        for await (const chunk of generator) {
            chunks.push(chunk)
        }

        // Verify stream chunks
        expect(chunks).toHaveLength(2) // One text chunk and one usage chunk
        expect(chunks[0]).toEqual({
            type: 'text',
            text: 'test response'
        })
        expect(chunks[1]).toEqual({
            type: 'usage',
            inputTokens: expect.any(Number),
            outputTokens: expect.any(Number),
            totalCost: expect.any(Number)
        })

        // Verify axios was called with correct parameters
        expect(axios.post).toHaveBeenCalledWith(
            'https://api.cerebras.ai/v1/chat/completions',
            expect.objectContaining({
                model: mockOptions.cerebrasModelId,
                messages: expect.arrayContaining([
                    { role: 'system', content: systemPrompt },
                    { role: 'user', content: 'test message' }
                ]),
                temperature: 0
            }),
            expect.objectContaining({
                headers: {
                    'Authorization': `Bearer ${mockOptions.cerebrasApiKey}`,
                    'Content-Type': 'application/json'
                }
            })
        )
    })
})
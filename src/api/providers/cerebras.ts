import axios from 'axios'
import { Anthropic } from '@anthropic-ai/sdk'
import { ApiHandler } from '../index'
import { ApiHandlerOptions, ModelInfo } from '../../shared/api'
import { ApiStream } from '../transform/stream'

export class CerebrasHandler implements ApiHandler {
    private apiKey: string
    private modelId: string
    private modelInfo: ModelInfo | undefined

    constructor(options: ApiHandlerOptions) {
        this.apiKey = options.cerebrasApiKey || ''
        this.modelId = options.cerebrasModelId || 'llama-3.3-70b' // Updated default model
        this.modelInfo = options.cerebrasModelInfo
    }

    getModel() {
        if (!this.modelId || !this.modelInfo) {
            throw new Error('Model information not provided')
        }

        return {
            id: this.modelId,
            info: this.modelInfo
        }
    }

    async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
        const allMessages = [
            { role: 'system', content: systemPrompt },
            ...messages
        ]

        try {
            const response = await axios.post(
                'https://api.cerebras.ai/v1/chat/completions',
                {
                    model: this.modelId,
                    messages: allMessages,
                    temperature: 0,
                    stream: false // Cerebras API doesn't support streaming yet
                },
                {
                    headers: {
                        'Authorization': `Bearer ${this.apiKey}`,
                        'Content-Type': 'application/json'
                    }
                }
            )

            const content = response.data.choices[0].message.content

            // Yield the text chunk
            yield {
                type: 'text',
                text: content
            }

            // Estimate token usage (can be refined based on actual token counting logic)
            const inputText = allMessages.map(m => m.content).join(' ')
            const inputTokens = Math.ceil(inputText.length / 4) // rough estimate
            const outputTokens = Math.ceil(content.length / 4) // rough estimate

            // Calculate cost based on model info
            const totalCost = 
                (inputTokens * (this.modelInfo?.inputPrice || 0)) +
                (outputTokens * (this.modelInfo?.outputPrice || 0))

            // Yield the usage chunk
            yield {
                type: 'usage',
                inputTokens,
                outputTokens,
                totalCost
            }

        } catch (error) {
            if (axios.isAxiosError(error)) {
                throw new Error(`Cerebras API error: ${error.response?.data?.error?.message || error.message}`)
            }
            throw error
        }
    }
}
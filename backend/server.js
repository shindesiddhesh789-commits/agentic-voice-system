const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const rateLimit = require('express-rate-limit');
const { OpenAI } = require('openai');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

if (!process.env.NVIDIA_API_KEY) {
  console.error('FATAL ERROR: NVIDIA_API_KEY is missing from .env file.');
  process.exit(1);
}

// Initialize OpenAI SDK to point to NVIDIA's NIM API
const openai = new OpenAI({
  baseURL: 'https://integrate.api.nvidia.com/v1',
  apiKey: process.env.NVIDIA_API_KEY
});

// Middleware
app.use(cors({ origin: process.env.FRONTEND_URL || '*' }));
app.use(express.json());

// Security: Rate Limiting (Prevents spam/abuse)
const apiLimiter = rateLimit({
  windowMs: 1 * 60 * 1000, // 1 minute
  max: 30,
  message: { error: 'Too many requests, please try again later.' }
});
app.use('/api/', apiLimiter);

// --- Simulated IoT Function Calling ---
const iotTools = [
  {
    type: 'function',
    function: {
      name: 'control_iot_device',
      description: 'Turn on or off a smart home device like lights, thermostat, or locks.',
      parameters: {
        type: 'object',
        properties: {
          device: { type: 'string', description: 'The name of the device (e.g., living room lights)' },
          action: { type: 'string', enum: ['turn_on', 'turn_off', 'check_status'] }
        },
        required: ['device', 'action']
      }
    }
  }
];

function executeIotAction(device, action) {
  console.log(`🔌 [NEMOTRON IOT] Executing: ${action} on ${device}`);
  if (action === 'check_status') return `The ${device} is currently functioning normally.`;
  return `Successfully performed ${action.replace('_', ' ')} on ${device}.`;
}

// --- API Endpoints ---

// 1. Main Chat Endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const { messages } = req.body;
    if (!messages || !Array.isArray(messages)) return res.status(400).json({ error: 'Invalid messages array' });

    // System Prompt for Voice Agent
    const apiMessages = [
      { role: 'system', content: 'You are an intelligent, helpful voice assistant. Keep answers concise, conversational, and direct for TTS output. Use tools if the user asks to control devices.' },
      ...messages
    ];

    // 1st Call to NVIDIA Nemotron
    let completion = await openai.chat.completions.create({
      model: 'nvidia/llama-3.3-nemotron-super-49b-v1.5',
      messages: apiMessages,
      tools: iotTools,
      temperature: 0.6,
      top_p: 0.95,
      max_tokens: 1024,
      frequency_penalty: 0,
      presence_penalty: 0
    });

    let message = completion.choices[0].message;

    // Handle IoT Tool Calls automatically
    if (message.tool_calls && message.tool_calls.length > 0) {
      apiMessages.push(message); 

      for (const toolCall of message.tool_calls) {
        if (toolCall.function.name === 'control_iot_device') {
          const args = JSON.parse(toolCall.function.arguments);
          const result = executeIotAction(args.device, args.action);
          
          apiMessages.push({
            role: 'tool',
            tool_call_id: toolCall.id,
            content: result
          });
        }
      }

      // 2nd Call to summarize the tool execution
      completion = await openai.chat.completions.create({
        model: 'nvidia/llama-3.3-nemotron-super-49b-v1.5',
        messages: apiMessages,
        temperature: 0.6,
        top_p: 0.95,
      });
      message = completion.choices[0].message;
    }

    res.json({ reply: message.content });
  } catch (error) {
    console.error('NVIDIA API Error:', error);
    res.status(500).json({ error: 'Failed to generate response.' });
  }
});

// 2. Smart Replies Endpoint
app.post('/api/smart-replies', async (req, res) => {
  try {
    const { messages } = req.body;
    const prompt = [
      ...messages,
      { role: 'system', content: 'Based on the conversation, suggest exactly 3 short follow-up actions or questions the user might want to say next. Output strictly as JSON: { "replies": ["str", "str", "str"] }' }
    ];

    const completion = await openai.chat.completions.create({
      model: 'nvidia/llama-3.3-nemotron-super-49b-v1.5',
      messages: prompt,
      response_format: { type: "json_object" },
      temperature: 0.3
    });

    const result = JSON.parse(completion.choices[0].message.content);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: 'Failed to generate replies.', replies: [] });
  }
});

// 3. Extract Tasks Endpoint
app.post('/api/extract-tasks', async (req, res) => {
  try {
    const { messages } = req.body;
    const prompt = [
      ...messages,
      { role: 'system', content: 'Extract any promised action items, tasks, or to-dos from this conversation. Return strictly as JSON: { "tasks": ["str", "str"] }. If none, return empty array.' }
    ];

    const completion = await openai.chat.completions.create({
      model: 'nvidia/llama-3.3-nemotron-super-49b-v1.5',
      messages: prompt,
      response_format: { type: "json_object" },
      temperature: 0.1
    });

    const result = JSON.parse(completion.choices[0].message.content);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: 'Failed to extract tasks.', tasks: [] });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 NVIDIA Nemotron Voice Backend running on http://localhost:${PORT}`);
});
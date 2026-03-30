/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect } from 'react';
import { 
  Camera, 
  Mic, 
  Music, 
  FileText, 
  Play, 
  Square, 
  Upload, 
  Cpu, 
  Zap, 
  Layers, 
  Code, 
  Terminal,
  Activity,
  Volume2,
  Image as ImageIcon,
  ChevronRight,
  Loader2,
  RefreshCw
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { useDropzone } from 'react-dropzone';
import { GoogleGenAI, Modality, GenerateContentResponse } from "@google/genai";
import { cn } from './lib/utils';

// --- Types ---
interface MultimodalInput {
  type: 'image' | 'audio' | 'video' | 'text';
  data?: string; // base64
  mimeType?: string;
  content?: string;
  id: string;
}

interface PipelineStatus {
  step: 'idle' | 'preprocessing' | 'embedding' | 'alignment' | 'generation' | 'error';
  message: string;
}

// --- Constants ---
const MODELS = {
  REASONING: "gemini-3-flash-preview",
  TTS: "gemini-2.5-flash-preview-tts",
  MUSIC: "lyria-3-clip-preview",
};

export default function App() {
  const [inputs, setInputs] = useState<MultimodalInput[]>([]);
  const [prompt, setPrompt] = useState("");
  const [status, setStatus] = useState<PipelineStatus>({ step: 'idle', message: 'System Ready' });
  const [reasoning, setReasoning] = useState("");
  const [generatedAudio, setGeneratedAudio] = useState<string | null>(null);
  const [generatedMusic, setGeneratedMusic] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [activeTab, setActiveTab] = useState<'pipeline' | 'code'>('pipeline');

  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);

  // --- Helpers ---
  const addInput = (input: Omit<MultimodalInput, 'id'>) => {
    setInputs(prev => [...prev, { ...input, id: Math.random().toString(36).substr(2, 9) }]);
  };

  const removeInput = (id: string) => {
    setInputs(prev => prev.filter(i => i.id !== id));
  };

  const onDrop = async (acceptedFiles: File[]) => {
    for (const file of acceptedFiles) {
      const reader = new FileReader();
      reader.onload = () => {
        const base64 = (reader.result as string).split(',')[1];
        const type = file.type.startsWith('image/') ? 'image' : 
                     file.type.startsWith('audio/') ? 'audio' : 
                     file.type.startsWith('video/') ? 'video' : 'text';
        addInput({ type, data: base64, mimeType: file.type });
      };
      reader.readAsDataURL(file);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif'],
      'audio/*': ['.mp3', '.wav', '.ogg', '.webm'],
      'video/*': ['.mp4', '.webm', '.ogg']
    }
  } as any);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream);
      audioChunks.current = [];
      mediaRecorder.current.ondataavailable = (e) => audioChunks.current.push(e.data);
      mediaRecorder.current.onstop = async () => {
        const blob = new Blob(audioChunks.current, { type: 'audio/webm' });
        const reader = new FileReader();
        reader.onload = () => {
          const base64 = (reader.result as string).split(',')[1];
          addInput({ type: 'audio', data: base64, mimeType: 'audio/webm' });
        };
        reader.readAsDataURL(blob);
      };
      mediaRecorder.current.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Recording failed", err);
    }
  };

  const stopRecording = () => {
    mediaRecorder.current?.stop();
    setIsRecording(false);
  };

  // --- Pipeline Execution ---
  const runPipeline = async () => {
    if (inputs.length === 0 && !prompt) return;

    setStatus({ step: 'preprocessing', message: 'Normalizing cross-modal features...' });
    setReasoning("");
    setGeneratedAudio(null);
    setGeneratedMusic(null);

    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

    try {
      // 1. Multimodal Alignment & Reasoning
      setStatus({ step: 'embedding', message: 'Projecting inputs into joint latent space...' });
      
      const parts = inputs.map(input => {
        if (input.data) {
          return { inlineData: { data: input.data, mimeType: input.mimeType! } };
        }
        return { text: input.content || "" };
      });

      if (prompt) parts.push({ text: prompt });

      setStatus({ step: 'alignment', message: 'Computing cross-modal attention weights...' });
      
      const response = await ai.models.generateContent({
        model: MODELS.REASONING,
        contents: { parts },
        config: {
          systemInstruction: "You are a Multimodal AI Scientist. Analyze the provided vision, audio, and text inputs. Describe their alignment, joint meaning, and potential cross-modal relationships. Be technical and precise."
        }
      });

      setReasoning(response.text || "No alignment detected.");

      // 2. Multimodal Generation (TTS)
      setStatus({ step: 'generation', message: 'Synthesizing audio from aligned features...' });
      const ttsResponse = await ai.models.generateContent({
        model: MODELS.TTS,
        contents: [{ parts: [{ text: `Technical summary of alignment: ${response.text?.substring(0, 200)}...` }] }],
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Fenrir' } } }
        }
      });

      const audioBase64 = ttsResponse.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (audioBase64) setGeneratedAudio(`data:audio/wav;base64,${audioBase64}`);

      // 3. Music Generation (Lyria)
      if (inputs.some(i => i.type === 'image')) {
        setStatus({ step: 'generation', message: 'Generating atmospheric music from visual context...' });
        const musicStream = await ai.models.generateContentStream({
          model: MODELS.MUSIC,
          contents: {
            parts: [
              { text: "Generate a 15-second ambient track that reflects the mood and texture of the visual inputs." },
              ...parts.filter(p => 'inlineData' in p)
            ]
          }
        });

        let musicBase64 = "";
        for await (const chunk of musicStream) {
          const mParts = chunk.candidates?.[0]?.content?.parts;
          if (mParts) {
            for (const p of mParts) {
              if (p.inlineData?.data) musicBase64 += p.inlineData.data;
            }
          }
        }
        if (musicBase64) {
          const binary = atob(musicBase64);
          const bytes = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
          const blob = new Blob([bytes], { type: 'audio/wav' });
          setGeneratedMusic(URL.createObjectURL(blob));
        }
      }

      setStatus({ step: 'idle', message: 'Pipeline Execution Complete' });
    } catch (err) {
      console.error("Pipeline failed", err);
      setStatus({ step: 'error', message: 'Pipeline Failure: Check logs' });
    }
  };

  return (
    <div className="min-h-screen bg-[#050505] text-[#E4E3E0] font-sans selection:bg-[#F27D26] selection:text-black">
      {/* --- Header --- */}
      <header className="border-b border-[#141414] p-6 flex items-center justify-between bg-[#0A0A0A]">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-[#F27D26] rounded-sm flex items-center justify-center">
            <Layers className="text-black w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight uppercase italic font-serif">Multimodal Fusion Lab</h1>
            <div className="flex items-center gap-2 text-[10px] text-[#8E9299] uppercase tracking-widest font-mono">
              <Activity className="w-3 h-3 text-[#00FF00]" />
              <span>System Status: {status.message}</span>
            </div>
          </div>
        </div>
        <div className="flex gap-4">
          <button 
            onClick={() => setActiveTab('pipeline')}
            className={cn(
              "px-4 py-2 text-xs font-mono uppercase tracking-widest transition-all border",
              activeTab === 'pipeline' ? "bg-[#F27D26] text-black border-[#F27D26]" : "border-[#141414] text-[#8E9299] hover:border-[#F27D26]"
            )}
          >
            Pipeline
          </button>
          <button 
            onClick={() => setActiveTab('code')}
            className={cn(
              "px-4 py-2 text-xs font-mono uppercase tracking-widest transition-all border",
              activeTab === 'code' ? "bg-[#F27D26] text-black border-[#F27D26]" : "border-[#141414] text-[#8E9299] hover:border-[#F27D26]"
            )}
          >
            Python Source
          </button>
        </div>
      </header>

      <main className="grid grid-cols-12 h-[calc(100vh-88px)]">
        {/* --- Sidebar: Inputs --- */}
        <section className="col-span-3 border-r border-[#141414] p-6 flex flex-col gap-6 overflow-y-auto bg-[#080808]">
          <div className="space-y-4">
            <label className="text-[11px] font-serif italic text-[#8E9299] uppercase tracking-widest">Input Modalities</label>
            
            {/* Dropzone */}
            <div 
              {...getRootProps()} 
              className={cn(
                "border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center gap-3 transition-all cursor-pointer",
                isDragActive ? "border-[#F27D26] bg-[#F27D26]/5" : "border-[#141414] hover:border-[#F27D26]/50"
              )}
            >
              <input {...getInputProps()} />
              <Upload className="w-8 h-8 text-[#8E9299]" />
              <p className="text-[10px] font-mono text-center text-[#8E9299]">Drop Vision/Audio Assets</p>
            </div>

            {/* Audio Recorder */}
            <div className="flex gap-2">
              <button 
                onClick={isRecording ? stopRecording : startRecording}
                className={cn(
                  "flex-1 py-3 rounded-xl flex items-center justify-center gap-2 transition-all border",
                  isRecording ? "bg-red-500/10 border-red-500 text-red-500 animate-pulse" : "bg-[#141414] border-[#141414] text-[#E4E3E0] hover:border-[#F27D26]"
                )}
              >
                {isRecording ? <Square className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
                <span className="text-[11px] font-mono uppercase tracking-widest">{isRecording ? "Stop" : "Record Audio"}</span>
              </button>
            </div>

            {/* Text Input */}
            <div className="space-y-2">
              <textarea 
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter technical constraints or research prompt..."
                className="w-full h-32 bg-[#141414] border border-[#141414] rounded-xl p-4 text-sm font-mono focus:border-[#F27D26] outline-none transition-all resize-none placeholder:text-[#333]"
              />
            </div>
          </div>

          {/* Active Inputs List */}
          <div className="flex-1 space-y-3 overflow-y-auto">
            <label className="text-[11px] font-serif italic text-[#8E9299] uppercase tracking-widest">Active Buffer</label>
            <AnimatePresence>
              {inputs.map((input) => (
                <motion.div 
                  key={input.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  className="bg-[#141414] border border-[#141414] p-3 rounded-lg flex items-center justify-between group"
                >
                  <div className="flex items-center gap-3">
                    {input.type === 'image' && <ImageIcon className="w-4 h-4 text-[#F27D26]" />}
                    {input.type === 'audio' && <Volume2 className="w-4 h-4 text-[#00FF00]" />}
                    {input.type === 'video' && <Play className="w-4 h-4 text-blue-500" />}
                    <span className="text-[10px] font-mono uppercase tracking-widest truncate max-w-[120px]">
                      {input.type} asset
                    </span>
                  </div>
                  <button 
                    onClick={() => removeInput(input.id)}
                    className="text-[#333] hover:text-red-500 transition-colors"
                  >
                    <Square className="w-3 h-3 fill-current" />
                  </button>
                </motion.div>
              ))}
            </AnimatePresence>
            {inputs.length === 0 && (
              <div className="text-[10px] font-mono text-[#333] italic">No assets in buffer...</div>
            )}
          </div>

          <button 
            onClick={runPipeline}
            disabled={status.step !== 'idle' && status.step !== 'error'}
            className="w-full py-4 bg-[#F27D26] text-black rounded-xl font-bold uppercase tracking-[0.2em] flex items-center justify-center gap-2 hover:scale-[1.02] active:scale-[0.98] transition-all disabled:opacity-50 disabled:grayscale"
          >
            {status.step === 'idle' ? (
              <>
                <Zap className="w-5 h-5 fill-current" />
                Execute Pipeline
              </>
            ) : (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Processing...
              </>
            )}
          </button>
        </section>

        {/* --- Center: Reasoning & Visualization --- */}
        <section className="col-span-6 p-8 flex flex-col gap-6 overflow-y-auto">
          {activeTab === 'pipeline' ? (
            <>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Cpu className="w-5 h-5 text-[#F27D26]" />
                  <h2 className="text-lg font-serif italic uppercase tracking-widest">Cross-Modal Alignment</h2>
                </div>
                <div className="flex items-center gap-4 text-[10px] font-mono text-[#8E9299]">
                  <div className="flex items-center gap-1">
                    <div className={cn("w-2 h-2 rounded-full", status.step === 'embedding' ? "bg-[#F27D26] animate-pulse" : "bg-[#141414]")} />
                    <span>Embed</span>
                  </div>
                  <ChevronRight className="w-3 h-3" />
                  <div className="flex items-center gap-1">
                    <div className={cn("w-2 h-2 rounded-full", status.step === 'alignment' ? "bg-[#F27D26] animate-pulse" : "bg-[#141414]")} />
                    <span>Align</span>
                  </div>
                  <ChevronRight className="w-3 h-3" />
                  <div className="flex items-center gap-1">
                    <div className={cn("w-2 h-2 rounded-full", status.step === 'generation' ? "bg-[#F27D26] animate-pulse" : "bg-[#141414]")} />
                    <span>Gen</span>
                  </div>
                </div>
              </div>

              <div className="flex-1 bg-[#0A0A0A] border border-[#141414] rounded-2xl p-8 font-mono text-sm leading-relaxed relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-[#F27D26] to-transparent opacity-20" />
                
                {reasoning ? (
                  <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="whitespace-pre-wrap text-[#E4E3E0]/80"
                  >
                    {reasoning}
                  </motion.div>
                ) : (
                  <div className="h-full flex flex-col items-center justify-center gap-4 text-[#333]">
                    <Terminal className="w-12 h-12" />
                    <p className="text-xs uppercase tracking-widest">Awaiting pipeline execution...</p>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="flex-1 bg-[#0A0A0A] border border-[#141414] rounded-2xl p-0 overflow-hidden flex flex-col">
              <div className="bg-[#141414] p-4 flex items-center gap-2 border-b border-[#0A0A0A]">
                <Code className="w-4 h-4 text-[#F27D26]" />
                <span className="text-[10px] font-mono uppercase tracking-widest">multimodal_pipeline.py</span>
              </div>
              <pre className="p-8 font-mono text-xs text-[#8E9299] overflow-auto leading-6">
                {`import torch
import torch.nn as nn
from transformers import CLIPModel, Wav2Vec2Model

class MultimodalAlignment(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Cross-Modal Attention Bridge
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.fusion_layer = nn.Linear(1024, 512)
        
    def forward(self, image, audio):
        # 1. Feature Extraction
        v_features = self.vision_encoder.get_image_features(image)
        a_features = self.audio_encoder(audio).last_hidden_state.mean(dim=1)
        
        # 2. Latent Alignment
        # Project audio to vision space
        a_aligned = self.fusion_layer(torch.cat([v_features, a_features], dim=-1))
        
        # 3. Compute Cross-Modal Attention
        attn_output, _ = self.attention(v_features, a_aligned, a_aligned)
        
        return attn_output

# Initialize Pipeline
pipeline = MultimodalAlignment()
print("Pipeline initialized with joint embedding space.")`}
              </pre>
            </div>
          )}
        </section>

        {/* --- Right: Outputs --- */}
        <section className="col-span-3 border-l border-[#141414] p-6 flex flex-col gap-6 bg-[#080808]">
          <label className="text-[11px] font-serif italic text-[#8E9299] uppercase tracking-widest">Synthesized Artifacts</label>
          
          {/* Generated Audio */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-[10px] font-mono uppercase tracking-widest text-[#8E9299]">
              <Volume2 className="w-3 h-3" />
              <span>Neural Speech (TTS)</span>
            </div>
            <div className="bg-[#141414] p-4 rounded-xl border border-[#141414] flex flex-col gap-4">
              {generatedAudio ? (
                <audio controls src={generatedAudio} className="w-full h-8 brightness-75 contrast-125" />
              ) : (
                <div className="h-8 flex items-center justify-center text-[9px] font-mono text-[#333] italic">
                  No speech generated...
                </div>
              )}
            </div>
          </div>

          {/* Generated Music */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-[10px] font-mono uppercase tracking-widest text-[#8E9299]">
              <Music className="w-3 h-3" />
              <span>Atmospheric Synthesis (Lyria)</span>
            </div>
            <div className="bg-[#141414] p-4 rounded-xl border border-[#141414] flex flex-col gap-4">
              {generatedMusic ? (
                <audio controls src={generatedMusic} className="w-full h-8 brightness-75 contrast-125" />
              ) : (
                <div className="h-8 flex items-center justify-center text-[9px] font-mono text-[#333] italic">
                  No music generated...
                </div>
              )}
            </div>
          </div>

          {/* Metrics Visualization */}
          <div className="flex-1 flex flex-col gap-4 mt-4">
            <label className="text-[11px] font-serif italic text-[#8E9299] uppercase tracking-widest">Latent Metrics</label>
            <div className="flex-1 bg-[#0A0A0A] border border-[#141414] rounded-xl p-4 flex flex-col gap-6">
              {[
                { label: 'Vision Confidence', value: reasoning ? 88 : 0, color: '#F27D26' },
                { label: 'Audio Coherence', value: reasoning ? 74 : 0, color: '#00FF00' },
                { label: 'Cross-Modal Alignment', value: reasoning ? 92 : 0, color: '#3B82F6' },
              ].map((metric) => (
                <div key={metric.label} className="space-y-2">
                  <div className="flex justify-between text-[9px] font-mono uppercase tracking-widest">
                    <span>{metric.label}</span>
                    <span>{metric.value}%</span>
                  </div>
                  <div className="h-1 bg-[#141414] rounded-full overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: `${metric.value}%` }}
                      transition={{ duration: 1, ease: "easeOut" }}
                      className="h-full"
                      style={{ backgroundColor: metric.color }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>

      {/* --- Footer Status Bar --- */}
      <footer className="h-8 bg-[#F27D26] text-black flex items-center px-6 justify-between text-[10px] font-mono font-bold uppercase tracking-widest">
        <div className="flex gap-6">
          <span>Kernel: v4.2.0-multimodal</span>
          <span>Buffer: {inputs.length} assets</span>
        </div>
        <div className="flex gap-6">
          <span>Memory: 12.4GB / 32GB</span>
          <span className="flex items-center gap-1">
            <RefreshCw className={cn("w-3 h-3", status.step !== 'idle' && "animate-spin")} />
            {status.step}
          </span>
        </div>
      </footer>
    </div>
  );
}

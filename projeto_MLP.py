# -*- coding: utf-8 -*-
"""
Editor Spyder

Este é um arquivo de script temporário.
"""

import numpy as np
def sigmoid(net):
    return 1/( 1 + np.exp(-net))
    
def df_sigmoid(f_net):
    return f_net * ( 1-f_net )

def tanh(net):
    return 2/(1+np.exp(-2*net))-1

def df_tanh(f_net):
    return 1-f_net**2

def linear(net):
    return net

def df_linear(f_net):
    return 1

funcoes= {'sigmoid':(sigmoid, df_sigmoid),
          'tanh': (tanh, df_tanh),
          'linear': (linear, df_linear)
          }


class MLP():
    def __init__(self,input_length=2, hidden_length=2, 
                    output_lenght=1, activation_function_h= 'sigmoid',
                    activation_function_o= 'sigmoid'):
        
        self.input_length= input_length
        self.hidden_lenght= hidden_length
        self.output_lenght= output_lenght
        
        self.hidden= np.random.rand( hidden_length, input_length+1) -0.5
        self.output= np.random.rand( output_lenght, hidden_length+1) -0.5
        
        self.f_h, self.df_dnet_h= funcoes[activation_function_h]
        self.f_o, self.df_dnet_o= funcoes[activation_function_o]



    def forward(self, Xp):
        #Hidden layer
        net_bias= np.concatenate(( Xp,np.array([1]) ))   
        self.net_h_p= self.hidden @ net_bias        
        self.f_net_h_p= self.f_h(self.net_h_p)
        
        #Output layer
        net_bias= np.concatenate(( self.f_net_h_p, np.array([1]) ))
        self.net_o_p= self.output @ net_bias
        self.f_net_o_p= self.f_o( self.net_o_p)

    def treinamento(self, previsores, resultado, eta= 0.1, epocas=5000, gamma= 0.9):
        
        for e in range(epocas):
            squaredError=0 
            v_h=0
            v_o=0
            
            for p in range(len(previsores)):     
                
                Xp= previsores[p]
                Yp= resultado[p]
                
                #aplica uma linha de dados na mlp e pega o resultado
                self.forward(Xp)                
                Op= self.f_net_o_p
                
                #Calculando erro
                error= Yp - Op
                
                squaredError += sum(error**2)
                
                #Training output
                delta_o_p= error* self.df_dnet_o( self.f_net_o_p)               
                delta_o_p= np.array([delta_o_p])
                
                #Training hidden
                w_o_kj= self.output[:,:self.hidden_lenght]               
                # Formato de delta_o_p (2,), mas como ele nao tem coluna, a multiplicacao de matrizes 
                # considera o fomato (1,2)
                delta_h_p=  self.df_dnet_h( self.f_net_h_p) * (delta_o_p @ w_o_kj)
                
                
                
                #Training
               
                f_net_hp_bias= np.concatenate((self.f_net_h_p, np.array([1]) ))
                
                # Python nao realiza multiplicação de matriz com dimensao faltando
                # Converter para array, mas ela inverte de (3,) para (1,3)
                # Usar .T para transpor e deixar no formato original so que (3,1)
               
                
                f_net_hp_bias= np.array([f_net_hp_bias])               
                
                
                
                v_o=  v_o*gamma +  eta* (delta_o_p.T @ f_net_hp_bias)
                self.output= self.output+ v_o
                
                
                Xp_bias= np.concatenate((Xp, np.array([1]) ))
                Xp_bias= np.array([Xp_bias])
                
                
                v_h= v_h*gamma + eta*  (delta_h_p.T @ Xp_bias)  
                self.hidden= self.hidden + v_h    
                
            squaredError/=len(previsores) 
            print('Erro médio quadrado', squaredError)
  
    
  
    


"""
modelo= MLP(output_lenght=2, hidden_length=2,   activation_function_o='sigmoid',
            activation_function_h= 'tanh')
           
Xor= np.array(([0,0,0,0], [0,1,1,1], [1,0,1,1], [1,1,0,0]))
modelo.treinamento(previsores= Xor[:,0:2], resultado= Xor[:,2:]
                   , epocas= 10000, gamma=0.9)
modelo.forward(np.array([1,0]))
print(modelo.f_net_o_p)
"""

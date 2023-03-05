# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 13:11:04 2023

@author: gabri
"""

# -*- coding: utf-8 -*-


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
        self.hidden_lenght= []
        self.hidden_lenght.append( hidden_length)    
        self.output_lenght= output_lenght
        
        self.hidden=[]
        self.hidden.append( np.random.rand( hidden_length, input_length+1) -0.5)
        self.output= np.random.rand( output_lenght, hidden_length+1) -0.5
        
        self.f_h, self.df_dnet_h= funcoes[activation_function_h]
        self.f_o, self.df_dnet_o= funcoes[activation_function_o]
        
        
       
        
        
    def add_camadas(self,  n_neuronios):  
        
        n_pesos= self.hidden_lenght[-1]
        self.hidden.append( np.random.rand( n_neuronios, n_pesos+1) -0.5)
       
        self.hidden_lenght.append( n_neuronios)
        self.output= np.random.rand( self.output_lenght, n_neuronios+1) -0.5

    def forward(self, Xp):
        #Hidden layer        
        self.net_h_p= []
        self.f_net_h_p= []
        
        net_bias= np.concatenate(( Xp,np.array([1]) ))           
        self.net_h_p.append( self.hidden[0] @ net_bias )  
        self.f_net_h_p.append( self.f_h( self.net_h_p[0]))
        
        
        for camada, pesos in enumerate(self.hidden):
            if camada!= 0:
                net_bias= np.concatenate(( self.f_net_h_p[ camada-1 ], np.array([1]) ))    
                self.net_h_p.append( pesos @ net_bias )    
                self.f_net_h_p.append( self.f_h( self.net_h_p[ camada]  ))        
                
       
        #Output layer
        
        net_bias= np.concatenate(( self.f_net_h_p[ camada], np.array([1]) ))
       
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
                w_o_kj= self.output[:,:self.hidden_lenght[-1]]               
                # Formato de delta_o_p (2,), mas como ele nao tem coluna, a multiplicacao de matrizes 
                # considera o fomato (1,2)
                delta_h_p=[]
                delta_h_p.append( self.df_dnet_h( self.f_net_h_p[-1]) * (delta_o_p @ w_o_kj))
                
                # o delta consiste na multiplicacao @ do delta da camada anterior * os pesos 
                # da camada anterior
                # ou seja para uma rede input=2, hiden 2, 4 ,5 ouput= 2
                # o delta da camada h_4  é o delta da h_5* pesos da h_5
                # e por fim a multiplica df  termo a termo 
                for i in range(2,len(self.hidden)+1, +1):
                    pesos= self.hidden[-i+1]
                    delta_h_p.append( self.df_dnet_h(self.f_net_h_p[-i]) * (
                        delta_h_p[i-2] @ pesos[:,:-1] ))
                
                
                # delta h_p começa da ultima camada
                # entao inverti para ficar na ordem 
                #ou seja o delta da ultima camada oculta  no final da lista 
                delta_h_p.reverse()
                
                #Training
               
                f_net_hp_bias= np.concatenate((self.f_net_h_p[-1], np.array([1]) ))               
                f_net_hp_bias= np.array([f_net_hp_bias])               
                
                v_o=  v_o*gamma +  eta* (delta_o_p.T @ f_net_hp_bias)
                self.output= self.output+ v_o
                
                # os pesos são atualizados com o delta da camada @ o resultado da função na camada anterior
                #ou seja, os pesos de h_4, sao h_4+ eta* delta_h4 @ (f_net_h2 com bias)
                for i in range(1,len(self.hidden)):
                    f_net_hp_bias= np.concatenate((self.f_net_h_p[-i-1], np.array([1]) ))   
                    f_net_hp_bias= np.array([f_net_hp_bias])                    
                    self.hidden[-i]= self.hidden[-i]+ eta*( delta_h_p[-i].T@ f_net_hp_bias)
                   
                    
                
                
                Xp_bias= np.concatenate((Xp, np.array([1]) ))
                Xp_bias= np.array([Xp_bias])
                
                
                v_h= v_h*gamma + eta*  (delta_h_p[0].T @ Xp_bias)  
                self.hidden[0]= self.hidden[0] + v_h    
                
            squaredError/=len(previsores) 
            print('Erro médio quadrado', squaredError)
  
    
  
    
if __name__ == '__main__':   


    modelo= MLP(output_lenght=2, hidden_length=2,   activation_function_o='sigmoid',
                activation_function_h= 'tanh')
    
    modelo.add_camadas( 10)
    modelo.add_camadas( 5)
    modelo.add_camadas( 100)
   
    Xor= np.array(([0,0,0,0], [0,1,1,1], [1,0,1,1], [1,1,0,0]))
    modelo.treinamento(previsores= Xor[:,0:2], resultado= Xor[:,2:]
                      , epocas=5000 , gamma=0)
    modelo.forward(np.array([1,0]))
    print(modelo.f_net_o_p)
    
